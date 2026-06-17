"""Configuration dataclasses for PM v2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: str = "."
    ignore_pm: bool = True


@dataclass(frozen=True)
class OllamaConfig:
    enabled: bool = False
    host: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    timeout: int = 120


@dataclass(frozen=True)
class WatcherConfig:
    enabled: bool = True
    recursive: bool = True
    debounce_ms: int = 300
    ignore_patterns: tuple[str, ...] = field(default_factory=lambda: (
        "*.pyc", "*.tmp", ".git/*", "__pycache__/*",
    ))


@dataclass(frozen=True)
class FormatterConfig:
    enabled: bool = True
    auto_format: bool = True
    strip_comments: bool = True
    max_blank_lines: int = 2


@dataclass(frozen=True)
class GitConfig:
    enabled: bool = True
    branch: str = "main"
    remote: str = "origin"
    auto_commit: bool = False
    auto_push: bool = False
    fallback_commit_message: str = "auto: project update"


@dataclass(frozen=True)
class TasksConfig:
    default: str = "sync"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "[%(name)s] %(message)s"


@dataclass(frozen=True)
class ProjectSettings:
    version: int
    project: ProjectConfig
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    formatter: FormatterConfig = field(default_factory=FormatterConfig)
    git: GitConfig = field(default_factory=GitConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
