"""Module registry — keeps track of all registered PM modules."""

from __future__ import annotations

from .exceptions import ModuleAlreadyRegisteredError, ModuleNotFoundError
from .module import Module


class ModuleRegistry:
    """
    Central store for all active PM modules.

    Modules self-register via ProjectManager.load_modules().
    After registration they are accessible by name.
    """

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, module: Module) -> None:
        """Register *module*. Raises if the name is already taken."""
        if module.name in self._modules:
            raise ModuleAlreadyRegisteredError(
                f"Module already registered: {module.name!r}"
            )
        self._modules[module.name] = module

    def unregister(self, name: str) -> None:
        """Remove a module by name (no-op if absent)."""
        self._modules.pop(name, None)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Module:
        """Return the module registered under *name*."""
        try:
            return self._modules[name]
        except KeyError:
            raise ModuleNotFoundError(f"No module registered: {name!r}")

    def get_or_none(self, name: str) -> Module | None:
        """Return the module or None if not found."""
        return self._modules.get(name)

    def all(self) -> list[Module]:
        """Return all registered modules in registration order."""
        return list(self._modules.values())

    def list_modules(self) -> list[str]:
        """Return module names in registration order."""
        return list(self._modules.keys())

    # kept as alias for backwards compat / convenience
    names = list_modules

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        return f"<ModuleRegistry modules={self.list_modules()}>"
