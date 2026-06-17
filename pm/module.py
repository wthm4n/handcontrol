from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ProjectManager


class Module(ABC):
    name: str

    def setup(self, pm: "ProjectManager") -> None:
        """Called once during initialization. Store refs from pm here."""

    def shutdown(self) -> None:
        """Called on teardown. Release resources, close connections."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
