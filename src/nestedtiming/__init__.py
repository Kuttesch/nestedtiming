"""Public API for nestedtiming.

This module re-exports the primary public interfaces:

- ``timing``: Decorator enabling optional hierarchical timing.
- ``timer``: Context manager for defining timing regions.
- ``timed``: Sentinel used to activate timing.
- ``TimingNode``: Public representation of a timing tree node.

Import from this module rather than ``nestedtiming.core``.
"""

from .core import TimingNode, timed, timer, timing

__all__ = [ "timing", "timer", "timed", "TimingNode"]