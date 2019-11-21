"""
fwks_examples
=============

Code for basic experiments in ``fwks`` library.
"""

from fwks import installer
from fwks.tasks import Task

__all__ = ["SpeechSeparation", "Task"]

from .speech_separation import SpeechSeparation

# Example of conditional definition
# if installer.is_installed("speech"):
#    from .speech_separation import SpeechSeparation
