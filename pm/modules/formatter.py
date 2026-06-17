from __future__ import annotations
import ast
import hashlib
import io
import json
import logging
import os
import pathlib
import tokenize
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from ..manager import ProjectManager

from ..module import Module

log = logging.getLogger("pm.formatter")

MAX_FILE_SIZE = 5 * 1024 * 1024
IGNORED_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".tox", "node_modules",
})
BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".onnx", ".pt", ".pth", ".pkl", ".pickle", ".dll", ".so",
    ".exe", ".bin", ".dylib", ".pyc", ".pyd", ".whl",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".db", ".sqlite", ".sqlite3", ".pdf",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov",
})
ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin1"]


class PythonCommentRemover:
    extensions = [".py"]
    _MAX_BLANK_LINES = 2

    def remove_comments(self, source: str) -> str:
        if not source:
            return source
        lines = self._strip_comment_tokens(source)
        return self._collapse_blank_lines(lines)

    def _strip_comment_tokens(self, source: str) -> list[str]:
        source_bytes = source.encode("utf-8")
        readline = io.BytesIO(source_bytes).readline
        original_lines = source.splitlines(keepends=True)
        lines_padded = [""] + original_lines
        comment_spans: dict[int, list[tuple[int, int]]] = {}
        try:
            for tok in tokenize.tokenize(readline):
                if tok.type == tokenize.COMMENT:
                    row, col_start = tok.start
                    _, col_end = tok.end
                    comment_spans.setdefault(row, []).append((col_start, col_end))
        except tokenize.TokenError:
            return original_lines

        result: list[str] = []
        for lineno, line in enumerate(lines_padded):
            if lineno == 0:
                continue
            spans = comment_spans.get(lineno)
            if not spans:
                result.append(line)
                continue
            chars = list(line)
            for col_start, col_end in sorted(spans, reverse=True):
                del chars[col_start:col_end]
            cleaned = "".join(chars).rstrip()
            ending = "\r\n" if line.endswith("\r\n") else "\r" if line.endswith("\r") else "\n"
            result.append((cleaned + ending) if cleaned else ending)
        return result

    def _collapse_blank_lines(self, lines: list[str]) -> str:
        result: list[str] = []
        consecutive_blank = 0
        for line in lines:
            if line.strip() == "":
                consecutive_blank += 1
                if consecutive_blank <= self._MAX_BLANK_LINES:
                    result.append(line)
            else:
                consecutive_blank = 0
                result.append(line)
        while result and result[0].strip() == "":
            result.pop(0)
        return "".join(result)


_REMOVERS = {ext: PythonCommentRemover() for ext in PythonCommentRemover.extensions}
_CACHE_FILE = pathlib.Path(__file__).parent / ".fmt_cache.json"


def _load_cache() -> dict[str, str]:
    if not _CACHE_FILE.is_file():
        return {}
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    tmp = _CACHE_FILE.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        os.replace(tmp, _CACHE_FILE)
    except OSError as exc:
        log.warning("Could not save cache: %s", exc)


def _file_hash(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _is_binary(path: pathlib.Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def _discover(root: pathlib.Path) -> list[pathlib.Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
        for fname in filenames:
            fp = pathlib.Path(dirpath) / fname
            if fp.suffix.lower() not in _REMOVERS:
                continue
            if _is_binary(fp):
                continue
            try:
                if fp.stat().st_size > MAX_FILE_SIZE:
                    log.warning("Skipping oversized file: %s", fp)
                    continue
            except OSError:
                continue
            found.append(fp)
    return sorted(found)


def _read_source(path: pathlib.Path) -> Optional[tuple[str, str]]:
    for enc in ENCODING_FALLBACKS:
        try:
            return path.read_text(encoding=enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _atomic_write(path: pathlib.Path, content: str, encoding: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def run_formatter(root: pathlib.Path) -> dict:
    cache = _load_cache()
    files = _discover(root)
    ok = skipped = errors = 0

    for fp in files:
        key = str(fp)
        try:
            current_hash = _file_hash(fp)
        except OSError:
            errors += 1
            continue

        if cache.get(key) == current_hash:
            skipped += 1
            continue

        result = _read_source(fp)
        if result is None:
            errors += 1
            continue
        source, encoding = result

        remover = _REMOVERS[fp.suffix.lower()]
        try:
            cleaned = remover.remove_comments(source)
        except Exception as exc:
            log.error("Formatter error on %s: %s", fp, exc)
            errors += 1
            continue

        if cleaned == source:
            cache[key] = current_hash
            skipped += 1
            continue

        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            log.error("Validation failed %s: %s", fp, exc)
            errors += 1
            continue

        try:
            _atomic_write(fp, cleaned, encoding)
            cache[key] = _file_hash(fp)
            ok += 1
            log.info("[ok] %s", fp)
        except OSError as exc:
            log.error("Write failed %s: %s", fp, exc)
            errors += 1

    _save_cache(cache)
    return {"processed": ok, "unchanged": skipped, "errors": errors}


class FormatterModule(Module):
    name = "formatter"

    def setup(self, pm: "ProjectManager") -> None:
        self._cfg = pm.config.formatter
        self._root = pm.root

    def run(self) -> dict:
        log.info("Running formatter (provider=%s, line_length=%d)", self._cfg.provider, self._cfg.line_length)
        return run_formatter(self._root)
