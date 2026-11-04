"""Run unmodified ROS 2 agent scripts against the simulator, without ROS."""

from .shim import SimBackend, install, uninstall  # noqa: F401

__all__ = ["install", "uninstall", "SimBackend"]
