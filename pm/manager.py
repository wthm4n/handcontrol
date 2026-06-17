from __future__ import annotations

import logging
from pathlib import Path

from .config import RootFinder
from .exceptions import ModuleSetupError
from .loader import ConfigLoader
from .models import ProjectSettings
from .module import Module
from .registry import ModuleRegistry


class ProjectManager:
    """
    Central orchestrator. Drop pm/ into any project, create a .src file, and go.

    Lifecycle:
        pm = ProjectManager()
        pm.load()   # discover root → load config → load modules → init modules

    Or step by step:
        pm.discover_root()
        pm.load_config()
        pm.load_modules()
        pm.initialize_modules()
    """

    def __init__(self) -> None:
        self._root: Path | None = None
        self._config: ProjectSettings | None = None
        self._registry = ModuleRegistry()
        self._logger = logging.getLogger("pm")
        self._root_finder = RootFinder()
        self._config_loader = ConfigLoader()

    def load(self) -> None:
        self.discover_root()
        self.load_config()
        self.load_modules()
        self.initialize_modules()

    def discover_root(self, start: Path | None = None) -> None:
        self._root = self._root_finder.find(start)
        self._logger.debug("Project root: %s", self._root)

    def load_config(self) -> None:
        if self._root is None:
            raise RuntimeError("Call discover_root() first.")
        self._config = self._config_loader.load(self._root)
        self._logger.debug("Config loaded: version=%d", self._config.version)

    def load_modules(self) -> None:
        from .modules.formatter import FormatterModule
        from .modules.source_controller import SourceControllerModule

        self._registry.register(FormatterModule())
        self._registry.register(SourceControllerModule())
        self._logger.debug("Modules registered: %s", self._registry.names())

    def initialize_modules(self) -> None:
        for module in self._registry.all():
            try:
                module.setup(self)
                self._logger.debug("Module initialized: %s", module.name)
            except Exception as exc:
                raise ModuleSetupError(f"Module '{module.name}' failed setup: {exc}") from exc

    def shutdown(self) -> None:
        for module in reversed(self._registry.all()):
            try:
                module.shutdown()
            except Exception:
                self._logger.exception("Error shutting down: %s", module.name)

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
    def registry(self) -> ModuleRegistry:
        return self._registry

    @property
    def modules(self) -> list[Module]:
        return self._registry.all()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def __repr__(self) -> str:
        root = str(self._root) if self._root else "not discovered"
        return f"<ProjectManager root={root!r} modules={self._registry.names()}>"
