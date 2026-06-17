import tomllib
from pathlib import Path

from .exceptions import ConfigNotFoundError, ConfigParseError, ConfigValidationError
from .models import FormatterConfig, ProjectConfig, ProjectSettings, SourceControllerConfig

CONFIG_FILENAME = ".src"


class ConfigLoader:
    def load(self, project_root: Path) -> ProjectSettings:
        raw = self._read(project_root)
        self._validate(raw)
        return self._build(raw)

    def _read(self, project_root: Path) -> dict:
        config_path = project_root / CONFIG_FILENAME
        if not config_path.exists():
            raise ConfigNotFoundError(f"Config file not found: {config_path}")
        try:
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigParseError(f"Failed to parse {config_path}: {exc}") from exc

    def _validate(self, raw: dict) -> None:
        errors: list[str] = []
        if "version" not in raw:
            errors.append("Missing required field: version")
        if "project" not in raw:
            errors.append("Missing required section: [project]")
        elif "name" not in raw.get("project", {}):
            errors.append("Missing required field: project.name")
        sc = raw.get("source_controller", {})
        if "source_controller" in raw and "enabled" not in sc:
            errors.append("Missing required field: source_controller.enabled")
        if "source_controller" in raw and sc.get("enabled") and "branch" not in sc:
            errors.append("Missing required field: source_controller.branch")
        if errors:
            raise ConfigValidationError(errors)

    def _build(self, raw: dict) -> ProjectSettings:
        project_raw = raw.get("project", {})
        fmt_raw = raw.get("formatter", {})
        sc_raw = raw.get("source_controller", {})
        return ProjectSettings(
            version=raw["version"],
            project=ProjectConfig(
                name=project_raw["name"],
                root=project_raw.get("root", "."),
            ),
            formatter=FormatterConfig(
                provider=fmt_raw.get("provider", "black"),
                line_length=fmt_raw.get("line_length", 88),
            ),
            source_controller=SourceControllerConfig(
                enabled=sc_raw.get("enabled", True),
                branch=sc_raw.get("branch", "main"),
                remote=sc_raw.get("remote", "origin"),
            ),
        )
