"""
pm panel — run from anywhere inside your project.

Usage:
    python -m pm              # interactive panel
    python -m pm status       # git status
    python -m pm commit       # stage + commit prompt
    python -m pm push         # push to remote
    python -m pm sync         # stage + commit + push (prompts for message)
    python -m pm fmt          # run comment formatter
    python -m pm log          # recent commits
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(levelname)s | %(message)s")

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def _header(title: str) -> None:
    print()
    print(_c(f"  {title}", BOLD, CYAN))
    print(_c("  " + "─" * (len(title) + 2), DIM))


def _ok(msg: str) -> None:
    print(_c(f"  ✓ {msg}", GREEN))


def _err(msg: str) -> None:
    print(_c(f"  ✗ {msg}", RED))


def _info(msg: str) -> None:
    print(_c(f"  · {msg}", DIM))


def _prompt(msg: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(_c(f"  → {msg}{hint}: ", YELLOW)).strip()
        return val or default
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)


def _load_pm():
    from pm import ProjectManager
    pm = ProjectManager()
    pm.load()
    return pm


def cmd_status(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    sc = pm.registry.get("source_controller")
    st = sc.status()

    _header("Git Status")
    print(_c(f"  branch  ", DIM) + _c(st.branch, BOLD, CYAN))
    print(_c(f"  remote  ", DIM) + _c(f"{pm.config.source_controller.remote}/{pm.config.source_controller.branch}", DIM))

    if st.ahead or st.behind:
        arrow = f"↑{st.ahead}" if st.ahead else ""
        arrow += f" ↓{st.behind}" if st.behind else ""
        print(_c(f"  sync    ", DIM) + _c(arrow, YELLOW))

    if not st.has_changes:
        print(_c("  nothing to commit, working tree clean", DIM))
    else:
        if st.staged:
            print()
            print(_c("  staged:", GREEN))
            for f in st.staged:
                print(_c(f"    + {f}", GREEN))
        if st.unstaged:
            print()
            print(_c("  modified (not staged):", YELLOW))
            for f in st.unstaged:
                print(_c(f"    ~ {f}", YELLOW))
        if st.untracked:
            print()
            print(_c("  untracked:", DIM))
            for f in st.untracked:
                print(_c(f"    ? {f}", DIM))
    print()


def cmd_commit(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    sc = pm.registry.get("source_controller")
    st = sc.status()

    _header("Commit")

    if not st.has_changes:
        _info("Nothing to commit.")
        print()
        return

    if not st.has_staged:
        print(_c("  No staged files. Stage all changes?", YELLOW))
        choice = _prompt("y/n", "y")
        if choice.lower() != "y":
            _info("Nothing staged, aborting.")
            print()
            return
        ok, msg = sc.stage_all()
        if not ok:
            _err(f"Stage failed: {msg}")
            print()
            return
        _ok("Staged all changes.")

    msg = _prompt("Commit message")
    if not msg:
        _err("Commit message is required.")
        print()
        return

    ok, out = sc.commit(msg)
    if ok:
        _ok(f"Committed: {out.splitlines()[0] if out else msg}")
    else:
        _err(f"Commit failed: {out}")
    print()


def cmd_push(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    sc = pm.registry.get("source_controller")

    _header("Push")
    ok, msg = sc.push()
    if ok:
        _ok(msg or "Pushed.")
    else:
        _err(f"Push failed: {msg}")
    print()


def cmd_sync(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    sc = pm.registry.get("source_controller")
    st = sc.status()

    _header("Sync  (stage → commit → push)")

    if not st.has_changes:
        _info("Nothing to commit.")
        print()
        return

    msg = _prompt("Commit message")
    if not msg:
        _err("Commit message is required.")
        print()
        return

    steps = sc.stage_commit_push(msg)
    for step, ok, out in steps:
        label = step.capitalize().ljust(7)
        if ok:
            _ok(f"{label} {out.splitlines()[0] if out else '✓'}")
        else:
            _err(f"{label} {out}")
            break
    print()


def cmd_fmt(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    fmt = pm.registry.get("formatter")

    _header("Formatter")
    result = fmt.run()
    _ok(f"processed  {result['processed']}")
    _info(f"unchanged  {result['unchanged']}")
    if result["errors"]:
        _err(f"errors     {result['errors']}")
    print()


def cmd_log(pm=None) -> None:
    if pm is None:
        pm = _load_pm()
    sc = pm.registry.get("source_controller")

    _header("Recent Commits")
    entries = sc.log(10)
    if not entries:
        _info("No commits found.")
    for e in entries:
        hash_str  = _c(e["hash"], CYAN)
        when_str  = _c(e["when"].ljust(14), DIM)
        author    = _c(e["author"].ljust(16), DIM)
        message   = e["message"]
        print(f"  {hash_str}  {when_str}  {author}  {message}")
    print()


def cmd_panel(pm) -> None:
    """Interactive panel loop."""
    sc = pm.registry.get("source_controller")
    st = sc.status()

    while True:
        _header(f"pm panel  —  {pm.config.project.name}")
        print(_c(f"  branch: {st.branch}", DIM))

        if st.has_changes:
            counts = []
            if st.staged:
                counts.append(_c(f"{len(st.staged)} staged", GREEN))
            if st.unstaged:
                counts.append(_c(f"{len(st.unstaged)} modified", YELLOW))
            if st.untracked:
                counts.append(_c(f"{len(st.untracked)} untracked", DIM))
            print(f"  {', '.join(counts)}")
        else:
            print(_c("  working tree clean", DIM))

        print()
        print(_c("  [s]", CYAN) + "  status")
        print(_c("  [c]", CYAN) + "  commit")
        print(_c("  [p]", CYAN) + "  push")
        print(_c("  [y]", CYAN) + "  sync  (stage + commit + push)")
        print(_c("  [f]", CYAN) + "  run formatter")
        print(_c("  [l]", CYAN) + "  log")
        print(_c("  [q]", CYAN) + "  quit")
        print()

        choice = _prompt("action").lower()

        if choice == "s":
            cmd_status(pm)
        elif choice == "c":
            cmd_commit(pm)
        elif choice == "p":
            cmd_push(pm)
        elif choice in ("y", "sync"):
            cmd_sync(pm)
        elif choice == "f":
            cmd_fmt(pm)
        elif choice == "l":
            cmd_log(pm)
        elif choice in ("q", "quit", "exit"):
            _info("bye")
            print()
            break
        else:
            _info(f"Unknown command: {choice!r}")


        st = sc.status()


COMMANDS = {
    "status": cmd_status,
    "commit": cmd_commit,
    "push":   cmd_push,
    "sync":   cmd_sync,
    "fmt":    cmd_fmt,
    "log":    cmd_log,
}


def main() -> None:
    args = sys.argv[1:]

    pm = _load_pm()

    if not args:
        cmd_panel(pm)
        return

    cmd = args[0].lower()
    if cmd in COMMANDS:
        COMMANDS[cmd](pm)
    else:
        print(_c(f"  Unknown command: {cmd!r}", RED))
        print(_c(f"  Available: {', '.join(COMMANDS)}", DIM))
        sys.exit(1)


if __name__ == "__main__":
    main()
