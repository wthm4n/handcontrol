from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: str = "."


@dataclass(frozen=True)
class FormatterConfig:
    provider: str = "black"
    line_length: int = 88


@dataclass(frozen=True)
class SourceControllerConfig:
    enabled: bool = True
    branch: str = "main"
    remote: str = "origin"


@dataclass(frozen=True)
class ProjectSettings:
    version: int
    project: ProjectConfig
    formatter: FormatterConfig = field(default_factory=FormatterConfig)
    source_controller: SourceControllerConfig = field(default_factory=SourceControllerConfig)
