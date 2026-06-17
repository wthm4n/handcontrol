"""TOML configuration loader."""

import logging
import tomllib
from pathlib import Path

from .exceptions import ConfigNotFoundError, ConfigParseError, ConfigValidationError
from .models import (
    FormatterConfig,
    GitConfig,
    LoggingConfig,
    OllamaConfig,
    ProjectConfig,
    ProjectSettings,
    TasksConfig,
    WatcherConfig,
)

CONFIG_FILENAME = ".src"
log = logging.getLogger("pm.config")


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
        git = raw.get("git", {})
        if git.get("enabled") and "branch" not in git:
            errors.append("Missing required field: git.branch (when git.enabled = true)")
        if errors:
            raise ConfigValidationError(errors)

    def _build(self, raw: dict) -> ProjectSettings:
        p = raw.get("project", {})
        o = raw.get("ollama", {})
        w = raw.get("watcher", {})
        f = raw.get("formatter", {})
        g = raw.get("git", {})
        t = raw.get("tasks", {})
        ll = raw.get("logging", {})

        if "commit_message" in g and "fallback_commit_message" not in g:
            log.warning(
                "[config] [git].commit_message is deprecated — rename it to "
                "[git].fallback_commit_message (it's now only used when AI "
                "generation is disabled or fails)."
            )

        return ProjectSettings(
            version=raw["version"],
            project=ProjectConfig(
                name=p["name"],
                root=p.get("root", "."),
                ignore_pm=p.get("ignore_pm", True),
            ),
            ollama=OllamaConfig(
                enabled=o.get("enabled", False),
                host=o.get("host", "http://localhost:11434"),
                model=o.get("model", "qwen2.5-coder:7b-instruct-q4_K_M"),
                timeout=o.get("timeout", 120),
            ),
            watcher=WatcherConfig(
                enabled=w.get("enabled", True),
                recursive=w.get("recursive", True),
                debounce_ms=w.get("debounce_ms", 300),
            ),
            formatter=FormatterConfig(
                enabled=f.get("enabled", True),
                auto_format=f.get("auto_format", True),
                strip_comments=f.get("strip_comments", True),
                max_blank_lines=f.get("max_blank_lines", 2),
            ),
            git=GitConfig(
                enabled=g.get("enabled", True),
                branch=g.get("branch", "main"),
                remote=g.get("remote", "origin"),
                auto_commit=g.get("auto_commit", False),
                auto_push=g.get("auto_push", False),
                fallback_commit_message=g.get(
                    "fallback_commit_message",
                    g.get("commit_message", "auto: project update"),
                ),
            ),
            tasks=TasksConfig(
                default=t.get("default", "sync"),
            ),
            logging=LoggingConfig(
                level=ll.get("level", "INFO"),
                format=ll.get("format", "[%(name)s] %(message)s"),
            ),
        )
