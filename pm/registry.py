from __future__ import annotations
from .exceptions import ModuleNotFoundError
from .module import Module


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    def register(self, module: Module) -> None:
        if module.name in self._modules:
            raise ValueError(f"Module already registered: {module.name!r}")
        self._modules[module.name] = module

    def get(self, name: str) -> Module:
        try:
            return self._modules[name]
        except KeyError:
            raise ModuleNotFoundError(f"No module registered: {name!r}")

    def all(self) -> list[Module]:
        return list(self._modules.values())

    def names(self) -> list[str]:
        return list(self._modules.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __repr__(self) -> str:
        return f"<ModuleRegistry modules={self.names()}>"
