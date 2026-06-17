from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager import ProjectManager

from ..module import Module

log = logging.getLogger("pm.source_controller")


@dataclass
class GitStatus:
    branch: str
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    ahead: int
    behind: int

    @property
    def has_changes(self) -> bool:
        return bool(self.staged or self.unstaged or self.untracked)

    @property
    def has_staged(self) -> bool:
        return bool(self.staged)


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


class SourceControllerModule(Module):
    name = "source_controller"

    def setup(self, pm: "ProjectManager") -> None:
        self._cfg = pm.config.source_controller
        self._root = pm.root


    def status(self) -> GitStatus:
        branch = self._current_branch()
        staged, unstaged, untracked = self._file_status()
        ahead, behind = self._ahead_behind()
        return GitStatus(
            branch=branch,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            ahead=ahead,
            behind=behind,
        )

    def _current_branch(self) -> str:
        code, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], self._root)
        return out if code == 0 else "unknown"

    def _file_status(self) -> tuple[list[str], list[str], list[str]]:
        code, out, _ = _run(["git", "status", "--porcelain"], self._root)
        if code != 0 or not out:
            return [], [], []

        staged, unstaged, untracked = [], [], []
        for line in out.splitlines():
            if len(line) < 3:
                continue
            xy = line[:2]
            filepath = line[3:]
            x, y = xy[0], xy[1]

            if xy == "??":
                untracked.append(filepath)
            else:
                if x != " " and x != "?":
                    staged.append(filepath)
                if y != " " and y != "?":
                    unstaged.append(filepath)
        return staged, unstaged, untracked

    def _ahead_behind(self) -> tuple[int, int]:
        code, out, _ = _run(
            ["git", "rev-list", "--left-right", "--count",
             f"HEAD...{self._cfg.remote}/{self._cfg.branch}"],
            self._root,
        )
        if code != 0 or not out:
            return 0, 0
        parts = out.split()
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 0, 0


    def stage_all(self) -> tuple[bool, str]:
        code, _, err = _run(["git", "add", "-A"], self._root)
        if code != 0:
            return False, err
        return True, "All changes staged."

    def stage_files(self, files: list[str]) -> tuple[bool, str]:
        if not files:
            return False, "No files provided."
        code, _, err = _run(["git", "add", "--"] + files, self._root)
        if code != 0:
            return False, err
        return True, f"Staged {len(files)} file(s)."

    def unstage_all(self) -> tuple[bool, str]:
        code, _, err = _run(["git", "reset", "HEAD"], self._root)
        if code != 0:
            return False, err
        return True, "All changes unstaged."

    def commit(self, message: str) -> tuple[bool, str]:
        if not message.strip():
            return False, "Commit message cannot be empty."
        st = self.status()
        if not st.has_staged:
            return False, "Nothing staged to commit."
        code, out, err = _run(["git", "commit", "-m", message], self._root)
        if code != 0:
            return False, err
        return True, out

    def push(self) -> tuple[bool, str]:
        code, out, err = _run(
            ["git", "push", self._cfg.remote, self._cfg.branch],
            self._root,
        )
        if code != 0:
            return False, err
        return True, out or f"Pushed to {self._cfg.remote}/{self._cfg.branch}."

    def stage_commit_push(self, message: str) -> list[tuple[str, bool, str]]:
        """One-shot: stage all → commit → push. Returns log of each step."""
        steps = []

        ok, msg = self.stage_all()
        steps.append(("stage", ok, msg))
        if not ok:
            return steps

        ok, msg = self.commit(message)
        steps.append(("commit", ok, msg))
        if not ok:
            return steps

        ok, msg = self.push()
        steps.append(("push", ok, msg))
        return steps

    def pull(self) -> tuple[bool, str]:
        code, out, err = _run(["git", "pull", self._cfg.remote, self._cfg.branch], self._root)
        if code != 0:
            return False, err
        return True, out or "Already up to date."

    def log(self, n: int = 10) -> list[dict]:
        code, out, _ = _run(
            ["git", "log", f"-{n}", "--pretty=format:%H|%an|%ar|%s"],
            self._root,
        )
        if code != 0 or not out:
            return []
        entries = []
        for line in out.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append({
                    "hash": parts[0][:7],
                    "author": parts[1],
                    "when": parts[2],
                    "message": parts[3],
                })
        return entries
