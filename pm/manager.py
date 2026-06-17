"""
ProjectManager — the central coordinator for PM v2.

PM owns everything. Modules never communicate directly with each other;
they communicate exclusively through pm.events (the EventBus).

Lifecycle:
    pm = ProjectManager()
    pm.load()           # discover → config → modules → init → start

    # Or step by step:
    pm.discover_root()
    pm.load_config()
    pm.load_modules()
    pm.initialize_modules()
    pm.start_modules()

Convenience properties:
    pm.git              → GitModule
    pm.ai               → AIModule
    pm.formatter        → FormatterModule
    pm.watcher          → WatcherModule
    pm.tasks            → TaskRunner
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import RootFinder
from .eventbus import EventBus, Events
from .exceptions import ModuleSetupError
from .loader import ConfigLoader
from .models import ProjectSettings
from .module import Module
from .registry import ModuleRegistry
from .scheduler import Scheduler


class ProjectManager:
    """Central orchestrator. Drop pm/ into any project, create a .src file."""

    def __init__(self) -> None:
        self._root: Path | None = None
        self._config: ProjectSettings | None = None
        self._registry = ModuleRegistry()
        self._events = EventBus()
        self._scheduler = Scheduler()
        self._root_finder = RootFinder()
        self._config_loader = ConfigLoader()
        self._logger = logging.getLogger("pm")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Full bootstrap in one call."""
        self.discover_root()
        self.load_config()
        self._configure_logging()
        self.load_modules()
        self.initialize_modules()
        self.start_modules()
        self._events.emit(Events.PM_LOADED)

    def discover_root(self, start: Path | None = None) -> None:
        self._root = self._root_finder.find(start)
        self._logger.debug("Project root: %s", self._root)

    def load_config(self) -> None:
        if self._root is None:
            raise RuntimeError("Call discover_root() first.")
        self._config = self._config_loader.load(self._root)
        self._logger.debug("Config loaded: version=%d", self._config.version)

    def load_modules(self) -> None:
        """Import and register all built-in modules."""
        from .modules.watcher.module import WatcherModule
        from .modules.formatter.module import FormatterModule
        from .modules.git.module import GitModule
        from .modules.ai.module import AIModule
        from .tasks.runner import TaskRunner

        self._registry.register(WatcherModule())
        self._registry.register(FormatterModule())
        self._registry.register(GitModule())
        self._registry.register(AIModule())
        self._registry.register(TaskRunner())

        self._logger.debug("Modules registered: %s", self._registry.list_modules())

    def initialize_modules(self) -> None:
        for module in self._registry.all():
            try:
                module.setup(self)
                self._logger.debug("Module ready: %s", module.name)
            except Exception as exc:
                raise ModuleSetupError(
                    f"Module '{module.name}' failed setup: {exc}"
                ) from exc

    def start_modules(self) -> None:
        for module in self._registry.all():
            try:
                module.start()
            except Exception as exc:
                self._logger.warning("Module '%s' failed start: %s", module.name, exc)

    def shutdown(self) -> None:
        self._events.emit(Events.PM_SHUTDOWN)
        for module in reversed(self._registry.all()):
            try:
                module.shutdown()
            except Exception:
                self._logger.exception("Error shutting down module: %s", module.name)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Call discover_root() first.")
        return self._root

    @property
    def config(self) -> ProjectSettings:
        if self._config is None:
            raise RuntimeError("Call load_config() first.")
        return self._config

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def registry(self) -> ModuleRegistry:
        return self._registry

    @property
    def modules(self) -> list[Module]:
        return self._registry.all()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # Module shortcuts —————————————————————————————————————————————
    @property
    def git(self):
        from .modules.git.module import GitModule
        return self._registry.get("git")  # type: GitModule

    @property
    def ai(self):
        from .modules.ai.module import AIModule
        return self._registry.get("ai")  # type: AIModule

    @property
    def formatter(self):
        from .modules.formatter.module import FormatterModule
        return self._registry.get("formatter")  # type: FormatterModule

    @property
    def watcher(self):
        from .modules.watcher.module import WatcherModule
        return self._registry.get("watcher")  # type: WatcherModule

    @property
    def tasks(self):
        from .tasks.runner import TaskRunner
        return self._registry.get("tasks")  # type: TaskRunner

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        level = getattr(logging, self._config.logging.level.upper(), logging.INFO)
        fmt = self._config.logging.format
        logging.basicConfig(level=level, format=fmt, force=True)
        self._logger.setLevel(level)

    def __repr__(self) -> str:
        root = str(self._root) if self._root else "not discovered"
        return f"<ProjectManager root={root!r} modules={self._registry.list_modules()}>"
