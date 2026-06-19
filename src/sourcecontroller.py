"""
gitdaemon.py — Professional auto-commit & push daemon.

Features:
  - Watches the working tree for any file change (content hash, not mtime)
  - Batches all dirty files into a single atomic commit per cycle
  - Generates intelligent commit messages from git diffs via the Anthropic API
    (falls back to a deterministic rule-based generator when offline / no key)
  - Pushes every PUSH_INTERVAL seconds when commits are pending
  - Retry with exponential back-off on push failure
  - Full-featured Rich TUI: live stats, file watcher, commit log, event log
  - Graceful SIGINT / SIGTERM shutdown

Usage:
  python gitdaemon.py [repo_path]          # defaults to cwd
  ANTHROPIC_API_KEY=sk-... python gitdaemon.py
"""

from __future__ import annotations

import os
import sys
import time
import signal
import hashlib
import subprocess
import threading
import textwrap
from collections import deque
from pathlib import Path
from typing import Optional


try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


PUSH_INTERVAL   = 5
SCAN_INTERVAL   = 1
MAX_DIFF_CHARS  = 3_000
MAX_RETRIES     = 5
RETRY_BASE      = 2
LOG_SIZE        = 14
COMMIT_LOG_SIZE = 10
FILE_LOG_SIZE   = 12

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".next", ".nuxt",
    "dist", "build", "out", "venv", ".venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".cache", "coverage",
    ".DS_Store",
}

IGNORE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".class",
    ".log", ".lock", ".map",
}


_RULE_TABLE = [

    ("auth",        "fix(auth)",    "tighten authentication logic"),
    ("login",       "fix(auth)",    "improve login flow"),
    ("judge",       "feat(judge)",  "enhance execution engine"),
    ("problem",     "ui(problem)",  "refine problem rendering"),
    ("verdict",     "ui(verdict)",  "improve verdict display"),
    ("route",       "feat(router)", "update routing"),
    ("api",         "feat(api)",    "update API layer"),
    ("model",       "feat(model)",  "update data model"),
    ("schema",      "feat(db)",     "update schema"),
    ("migration",   "feat(db)",     "add migration"),
    ("test",        "test",         "update test suite"),
    ("spec",        "test",         "update specs"),
    ("config",      "chore(cfg)",   "update configuration"),
    ("docker",      "chore(ops)",   "update Docker setup"),
    ("ci",          "ci",           "update CI pipeline"),
    ("readme",      "docs",         "update README"),
    ("docs",        "docs",         "update documentation"),
    ("style",       "style",        "apply style updates"),
]

_EXT_MAP = {
    ".jsx": "ui",    ".tsx": "ui",    ".vue": "ui",
    ".css": "style", ".scss": "style", ".sass": "style",
    ".py":  "feat",  ".go":  "feat",  ".rs":  "feat",
    ".ts":  "feat",  ".js":  "feat",
    ".md":  "docs",  ".rst": "docs",
    ".yml": "chore", ".yaml": "chore", ".toml": "chore",
    ".sh":  "chore", ".env":  "chore",
}


def _rule_based_message(files: list[str]) -> str:
    """Fast, deterministic commit message from file names and extensions."""
    if len(files) == 1:
        f = files[0].lower()
        name = Path(f).name
        ext  = Path(f).suffix

        for keyword, prefix, action in _RULE_TABLE:
            if keyword in f:
                return f"{prefix}: {action}"

        scope = Path(f).stem
        tag   = _EXT_MAP.get(ext, "chore")
        return f"{tag}({scope}): update {name}"


    all_lower = " ".join(files).lower()
    for keyword, prefix, action in _RULE_TABLE:
        if keyword in all_lower:
            return f"{prefix}: update {len(files)} files"

    exts = {Path(f).suffix for f in files}
    if exts <= {".css", ".scss", ".sass"}:
        return f"style: update {len(files)} stylesheets"
    if exts <= {".jsx", ".tsx", ".vue"}:
        return f"ui: update {len(files)} components"
    if exts <= {".py"}:
        return f"refactor: update {len(files)} Python modules"
    if exts <= {".md", ".rst"}:
        return f"docs: update {len(files)} documents"

    return f"chore: update {len(files)} files"


def _ai_commit_message(files: list[str], diff: str, api_key: str) -> Optional[str]:
    """Ask Claude for a single-line conventional-commit message."""
    if not _ANTHROPIC_AVAILABLE or not api_key:
        return None

    file_list = "\n".join(f"  • {f}" for f in files[:20])
    prompt = textwrap.dedent(f"""
        You are a senior engineer writing a Git commit message.

        Changed files:
        {file_list}

        Diff (truncated to {MAX_DIFF_CHARS} chars):
        {diff[:MAX_DIFF_CHARS]}

        Rules:
        - Return ONLY the commit subject line, nothing else.
        - Use Conventional Commits format: type(scope): description
        - type: feat | fix | ui | refactor | style | test | docs | chore | ci
        - scope: the affected module/component (short)
        - description: imperative, ≤72 chars, lowercase
        - No trailing period. No quotes. No markdown.
    """).strip()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().strip('"').strip("'")
        if text and len(text) <= 100:
            return text
    except Exception:
        pass

    return None


def get_commit_message(files: list[str], diff: str, api_key: str) -> tuple[str, str]:
    """Returns (message, source) where source is 'ai' or 'rule'."""
    ai_msg = _ai_commit_message(files, diff, api_key)
    if ai_msg:
        return ai_msg, "ai"
    return _rule_based_message(files), "rule"


def _run(cmd: str, cwd: str) -> tuple[int, str, str]:
    r = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def git_status(cwd: str) -> list[str]:
    """Return list of dirty relative paths (skips ignored dirs/exts)."""
    rc, out, _ = _run("git status --porcelain=v1", cwd)
    if rc != 0:
        return []
    results = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"')

        parts = Path(path).parts
        if any(p in IGNORE_DIRS for p in parts):
            continue
        if Path(path).suffix in IGNORE_EXTS:
            continue
        results.append(path)
    return results


def git_diff(files: list[str], cwd: str) -> str:
    """Get combined diff for the given files (staged + unstaged)."""
    file_args = " ".join(f'"{f}"' for f in files[:30])
    _, staged,   _ = _run(f"git diff --cached {file_args}", cwd)
    _, unstaged, _ = _run(f"git diff {file_args}", cwd)
    combined = (staged + "\n" + unstaged).strip()

    if not combined:
        combined = "New/untracked files: " + ", ".join(files)
    return combined


def git_add_all(cwd: str) -> bool:
    rc, _, _ = _run("git add -A", cwd)
    return rc == 0


def git_commit(message: str, cwd: str) -> bool:
    rc, _, _ = _run(f'git commit -m "{message}"', cwd)
    return rc == 0


def git_push(cwd: str) -> tuple[bool, str]:
    rc, out, err = _run("git push", cwd)
    return rc == 0, (err or out)


def git_branch(cwd: str) -> str:
    _, out, _ = _run("git branch --show-current", cwd)
    return out or "detached"


def git_remote(cwd: str) -> str:
    _, out, _ = _run("git remote get-url origin", cwd)
    if out:

        out = out.replace("https://github.com/", "gh:")
        out = out.replace("git@github.com:", "gh:")
    return out or "(no remote)"


def file_hash(path: str) -> Optional[str]:
    try:
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class DaemonState:
    def __init__(self, repo: str, api_key: str):
        self.repo        = repo
        self.api_key     = api_key
        self.branch      = git_branch(repo)
        self.remote      = git_remote(repo)


        self.commit_count  = 0
        self.push_count    = 0
        self.ai_count      = 0
        self.fail_count    = 0
        self.start_time    = time.time()


        self.event_log     : deque[tuple[float, str, str]] = deque(maxlen=LOG_SIZE)
        self.commit_log    : deque[tuple[float, str, str]] = deque(maxlen=COMMIT_LOG_SIZE)
        self.watched_files : deque[tuple[str, str]]        = deque(maxlen=FILE_LOG_SIZE)


        self.pending_push  = False
        self.last_push_ts  = time.time()
        self.last_push_str = "—"
        self.retry_count   = 0


        self._hashes: dict[str, str] = {}


        self._lock = threading.Lock()

        self._stop = threading.Event()


    def log(self, msg: str, level: str = "info") -> None:
        self.event_log.appendleft((time.time(), level, msg))

    def uptime(self) -> str:
        s = int(time.time() - self.start_time)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()


    def cycle(self) -> None:
        with self._lock:
            self._scan_and_commit()
            self._maybe_push()

    def _scan_and_commit(self) -> None:
        dirty = git_status(self.repo)


        self.watched_files.clear()
        for f in dirty:
            p = os.path.join(self.repo, f)
            h = file_hash(p)
            self.watched_files.append((f, h or "deleted"))

        if not dirty:
            return


        changed = []
        for f in dirty:
            p = os.path.join(self.repo, f)
            h = file_hash(p)
            prior = self._hashes.get(f, "__NEW__")
            if h != prior:
                changed.append(f)

        if not changed:
            return


        if not git_add_all(self.repo):
            self.log("git add -A failed", "error")
            return


        diff = git_diff(changed, self.repo)


        msg, source = get_commit_message(changed, diff, self.api_key)


        if git_commit(msg, self.repo):
            self.commit_count += 1
            if source == "ai":
                self.ai_count += 1

            for f in changed:
                p = os.path.join(self.repo, f)
                self._hashes[f] = file_hash(p) or ""
            self.pending_push = True
            label = "🤖 ai" if source == "ai" else "📐 rule"
            self.commit_log.appendleft((time.time(), msg, label))
            self.log(f"Committed {len(changed)} file(s): {msg}", "commit")
        else:
            self.log("Commit failed (nothing to commit?)", "warn")

    def _maybe_push(self) -> None:
        if not self.pending_push:
            return

        now = time.time()
        wait = PUSH_INTERVAL * (RETRY_BASE ** self.retry_count)

        if now - self.last_push_ts < wait:
            return

        ok, detail = git_push(self.repo)

        if ok:
            self.push_count    += 1
            self.retry_count    = 0
            self.pending_push   = False
            self.last_push_ts   = now
            self.last_push_str  = time.strftime("%H:%M:%S")
            self.log("Pushed to remote ✓", "push")
        else:
            self.fail_count += 1
            self.retry_count = min(self.retry_count + 1, MAX_RETRIES)
            self.last_push_ts = now
            short = (detail or "unknown error")[:80]
            self.log(f"Push failed (retry {self.retry_count}): {short}", "error")


LEVEL_STYLE = {
    "info":   "dim white",
    "commit": "bold green",
    "push":   "bold bright_cyan",
    "warn":   "yellow",
    "error":  "bold red",
}

LEVEL_ICON = {
    "info":   "·",
    "commit": "●",
    "push":   "▲",
    "warn":   "⚠",
    "error":  "✖",
}


def _ts(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def build_header(st: DaemonState) -> Panel:
    t = Text()
    t.append("  ⬡ ", style="bold bright_cyan")
    t.append("GITDAEMON", style="bold white")
    t.append("  —  auto-commit · auto-push  ", style="dim")
    t.append(f"  uptime {st.uptime()}  ", style="bright_black")
    branch_style = "bold green"
    t.append(f"  ⎇  {st.branch}  ", style=branch_style)
    t.append(f"  {st.remote}", style="dim cyan")
    return Panel(Align.center(t), border_style="bright_cyan", padding=(0, 1))


def build_stats(st: DaemonState) -> Panel:
    g = Table.grid(expand=True, padding=(0, 2))
    g.add_column(justify="left",  no_wrap=True)
    g.add_column(justify="right", no_wrap=True)
    g.add_column(justify="left",  no_wrap=True)
    g.add_column(justify="right", no_wrap=True)

    def row(l1, v1, l2, v2):
        g.add_row(
            f"[bright_black]{l1}[/bright_black]",
            f"[bold white]{v1}[/bold white]",
            f"[bright_black]{l2}[/bright_black]",
            f"[bold white]{v2}[/bold white]",
        )

    row("COMMITS",  str(st.commit_count),
        "PUSHES",   str(st.push_count))
    row("AI MSGS",  str(st.ai_count),
        "FAILURES", f"[red]{st.fail_count}[/red]" if st.fail_count else "0")

    push_status = (
        "[yellow]PENDING[/yellow]"
        if st.pending_push else
        "[bright_black]IDLE[/bright_black]"
    )
    row("QUEUE",    push_status,
        "LAST PUSH", f"[cyan]{st.last_push_str}[/cyan]")

    return Panel(
        g,
        title="[bold white]STATS[/bold white]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def build_files(st: DaemonState) -> Panel:
    t = Table(expand=True, show_header=True, header_style="bright_black",
              border_style="bright_black", padding=(0, 1))
    t.add_column("FILE", style="white", no_wrap=False, ratio=4)
    t.add_column("HASH", style="bright_black", no_wrap=True, ratio=1)

    if st.watched_files:
        for f, h in list(st.watched_files):
            short_h = h[:8] if h and h != "deleted" else "[red]deleted[/red]"
            t.add_row(f, short_h)
    else:
        t.add_row("[bright_black]— clean working tree —[/bright_black]", "")

    return Panel(
        t,
        title="[bold yellow]DIRTY FILES[/bold yellow]",
        border_style="yellow",
        padding=(0, 0),
    )


def build_commits(st: DaemonState) -> Panel:
    t = Table(expand=True, show_header=False, border_style="bright_black",
              padding=(0, 1))
    t.add_column("TIME",   style="bright_black", no_wrap=True, width=8)
    t.add_column("MSG",    style="green",        no_wrap=False, ratio=5)
    t.add_column("SOURCE", style="bright_black", no_wrap=True, width=8)

    if st.commit_log:
        for ts, msg, source in list(st.commit_log):
            t.add_row(_ts(ts), msg, source)
    else:
        t.add_row("", "[bright_black]— no commits yet —[/bright_black]", "")

    return Panel(
        t,
        title="[bold green]COMMITS[/bold green]",
        border_style="green",
        padding=(0, 0),
    )


def build_log(st: DaemonState) -> Panel:
    t = Table(expand=True, show_header=False, border_style="bright_black",
              padding=(0, 1))
    t.add_column("TIME",  style="bright_black", no_wrap=True, width=8)
    t.add_column("ICON",  no_wrap=True, width=1)
    t.add_column("MSG",   no_wrap=False, ratio=6)

    if st.event_log:
        for ts, level, msg in list(st.event_log):
            style = LEVEL_STYLE.get(level, "white")
            icon  = LEVEL_ICON.get(level, "·")
            t.add_row(
                _ts(ts),
                f"[{style}]{icon}[/{style}]",
                f"[{style}]{msg}[/{style}]",
            )
    else:
        t.add_row("", "·", "[bright_black]waiting…[/bright_black]")

    return Panel(
        t,
        title="[bold magenta]EVENT LOG[/bold magenta]",
        border_style="magenta",
        padding=(0, 0),
    )


def build_footer() -> Panel:
    t = Text()
    t.append(" ^C ", style="bold black on bright_white")
    t.append(" stop   ", style="bright_black")
    t.append(" ^Z ", style="bold black on bright_white")
    t.append(" suspend   ", style="bright_black")
    t.append(f" push interval: {PUSH_INTERVAL}s   ", style="bright_black")
    t.append(f" scan interval: {SCAN_INTERVAL}s", style="bright_black")
    return Panel(Align.center(t), border_style="bright_black", padding=(0, 1))


def build_layout(st: DaemonState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",  size=3),
        Layout(name="body"),
        Layout(name="footer",  size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="stats",  size=8),
        Layout(name="files"),
    )
    layout["right"].split_column(
        Layout(name="commits"),
        Layout(name="log"),
    )

    layout["header"].update(build_header(st))
    layout["stats"].update(build_stats(st))
    layout["files"].update(build_files(st))
    layout["commits"].update(build_commits(st))
    layout["log"].update(build_log(st))
    layout["footer"].update(build_footer())

    return layout


def main() -> None:
    repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())

    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"[error] Not a git repository: {repo}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[warn] ANTHROPIC_API_KEY not set — using rule-based commit messages.",
              file=sys.stderr)

    state = DaemonState(repo=repo, api_key=api_key)
    state.log("Daemon started", "info")
    state.log(f"Watching {repo}", "info")

    if api_key and _ANTHROPIC_AVAILABLE:
        state.log("AI commit messages enabled (Claude Haiku)", "info")
    else:
        state.log("Rule-based commit messages active", "info")


    def _shutdown(sig, frame):
        state.log("Shutdown signal received", "warn")
        state.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    console = Console()

    with Live(
        build_layout(state),
        console=console,
        refresh_per_second=4,
        screen=True,
    ) as live:
        while not state.should_stop():
            try:
                state.cycle()
            except Exception as exc:
                state.log(f"{type(exc).__name__}: {exc}", "error")

            live.update(build_layout(state))
            time.sleep(SCAN_INTERVAL)


        live.update(build_layout(state))

    print("\n[gitdaemon] stopped cleanly.")


if __name__ == "__main__":
    main()