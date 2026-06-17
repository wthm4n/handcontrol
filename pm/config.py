"""Project root discovery."""

from pathlib import Path

from .exceptions import RootNotFoundError

CONFIG_FILENAME = ".src"


class RootFinder:
    """Walk up directory tree looking for a .src config file."""

    def find(self, start: Path | None = None) -> Path:
        current = (start or Path.cwd()).resolve()
        for directory in [current, *current.parents]:
            if (directory / CONFIG_FILENAME).exists():
                return directory
        raise RootNotFoundError(
            f"Could not find '{CONFIG_FILENAME}' in {current} or any parent directory."
        )
