class PMError(Exception):
    """Base exception for all Project Manager errors."""


class RootNotFoundError(PMError):
    """Raised when the project root cannot be located."""


class ConfigNotFoundError(PMError):
    """Raised when the .src config file is missing."""


class ConfigParseError(PMError):
    """Raised when the .src file cannot be parsed as TOML."""


class ConfigValidationError(PMError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


class ModuleError(PMError):
    """Base exception for module-related errors."""


class ModuleNotFoundError(ModuleError):
    """Raised when a requested module is not registered."""


class ModuleSetupError(ModuleError):
    """Raised when a module fails during setup."""
