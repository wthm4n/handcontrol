"""
PM — project automation framework.

    from pm import ProjectManager

    pm = ProjectManager()
    pm.load()
    pm.tasks.run("sync")
"""

from .manager import ProjectManager

__version__ = "2.0.0"
__all__ = ["ProjectManager", "__version__"]
