"""PM exception hierarchy."""


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


class ModuleAlreadyRegisteredError(ModuleError):
    """Raised when a module name collision occurs."""


class ModuleSetupError(ModuleError):
    """Raised when a module fails during setup."""


class TaskError(PMError):
    """Raised when a task fails."""


class TaskNotFoundError(TaskError):
    """Raised when a task name is not recognized."""


class GitError(PMError):
    """Raised when a git operation fails."""


class OllamaError(PMError):
    """Raised when Ollama integration fails."""


class WatcherError(PMError):
    """Raised when the file watcher encounters an error."""
