"""Base class for all PM modules."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ProjectManager


class Module(ABC):
    """
    Base class every PM module must inherit.

    Lifecycle:
        module.setup(pm)      # called once by ProjectManager.initialize_modules()
        module.shutdown()     # called on teardown in reverse registration order

    Modules communicate exclusively through pm.events (the EventBus).
    Modules MUST NOT import or call each other directly.
    """

    #: Each subclass must declare a unique lowercase name.
    name: str

    def setup(self, pm: "ProjectManager") -> None:
        """
        Initialise the module.

        Store references from pm here (root, config, events).
        Subscribe to events here.
        Do NOT start background threads here — override start() for that.
        """

    def start(self) -> None:
        """
        Start background activity (watchers, schedulers, etc.).

        Called after all modules have been set up.
        """

    def shutdown(self) -> None:
        """
        Tear down the module.

        Stop threads, close files, flush caches.
        Called in reverse registration order.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
