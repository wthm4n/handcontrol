"""
Demo module for testing the PM formatter.

This file intentionally contains:
- Module docstrings
- Class docstrings
- Function docstrings
- Inline comments
- Block comments
- Type hints
- Dataclasses
- Enums
- Decorators

The formatter should remove comments while preserving
valid Python syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


APP_VERSION = "1.0.0"

def version() -> str:
    """
    Return the current application version.
    """

    return APP_VERSION

def greet(name: str) -> str:
    """
    Return a greeting message for *name*.
    """

    return f"Hello, {name}!"


class Status(Enum):
    """Represents task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def showTime() -> None:
    """
    Print the current time.
    """

    from datetime import datetime

    now = datetime.now()

    print(f"Current time: {now}")

@dataclass
class Task:
    """
    Represents a task.

    Attributes:
        name: Task name.
        status: Current task status.
        priority: Task priority.
    """

    name: str
    status: Status = Status.PENDING
    priority: int = 1


def timer(func: Callable):
    """
    Simple decorator.
    """

    def wrapper(*args, **kwargs):

        return func(*args, **kwargs)

    return wrapper


class TaskManager:
    """
    Manages tasks.
    """

    def __init__(self) -> None:

        self.tasks: list[Task] = []

    def add_task(self, task: Task) -> None:
        """
        Add a task.
        """

        if not task.name:
            raise ValueError("Task name cannot be empty")

        self.tasks.append(task)

    def get_task(self, name: str) -> Task | None:
        """
        Find task by name.
        """

        for task in self.tasks:
            if task.name == name:
                return task

        return None

    def remove_task(self, name: str) -> bool:
        """
        Remove a task.
        """

        for task in list(self.tasks):
            if task.name == name:
                self.tasks.remove(task)
                return True

        return False

    def list_tasks(self) -> list[Task]:
        """
        Return all tasks.
        """

        return list(self.tasks)

    @timer
    def process_tasks(self) -> None:
        """
        Process all pending tasks.
        """

        for task in self.tasks:

            if task.status == Status.COMPLETED:
                continue

            task.status = Status.RUNNING

            try:

                result = self._execute_task(task)

                if result:
                    task.status = Status.COMPLETED
                else:
                    task.status = Status.FAILED

            except Exception:
                task.status = Status.FAILED

    def _execute_task(self, task: Task) -> bool:
        """
        Internal execution routine.
        """

        def validate() -> bool:

            return bool(task.name)

        return validate()


def load_project(root: Path) -> dict:
    """
    Load a fake project configuration.
    """

    config = {
        "name": "Demo Project",
        "root": str(root),
        "debug": True,
    }

    return config


def calculate_statistics(values: list[int]) -> dict:
    """
    Calculate basic statistics.
    """

    if not values:
        return {
            "count": 0,
            "sum": 0,
            "average": 0,
        }

    total = sum(values)

    average = total / len(values)

    return {
        "count": len(values),
        "sum": total,
        "average": average,
    }


def main() -> None:
    """
    Program entry point.
    """

    manager = TaskManager()

    manager.add_task(Task("Build"))
    manager.add_task(Task("Test"))
    manager.add_task(Task("Deploy"))

    manager.process_tasks()

    for task in manager.list_tasks():
        print(f"{task.name}: {task.status.value}")

    stats = calculate_statistics([1, 2, 3, 4, 5])

    print(stats)


def run():
    """
    Run the main function.
    """

    main()


if __name__ == "__main__":

    main()
