"""
FWKS installer module
Makes sure everything is distributed and compiled for FWKS to use.

FWKS has some binary dependencies and in order to fully use it, the dependencies should
be installed in proper way. Installer is a tool that may help in the process (it is however not automatic)
"""

from fwks.installer.meta import Dependency


def is_installed(dependency):
    """
    Verify whether given dependency is installed.
    """
    return Dependency.all_instances()[dependency].is_installed()


def all_dependencies():
    """
    Returns all registered dependencies in a form of a list of names
    """
    return Dependency.all_instances().keys()


def dependency_installer(dependency):
    """
    Returns a callback that is used to install dependency. Each callback
    has only optional and keyword inputs.
    """
    return Dependency.all_instances()[dependency].installer
