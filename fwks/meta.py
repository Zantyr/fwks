"""
fwks.meta
=========

This module is home to helper classes, but also hosts the main configuration management code for the library
"""


__all__ = ["Watcher", "LocalRepo"]


class Watcher(type):
    """
    Based on https://stackoverflow.com/questions/18126552
    """

    count = 0
    instances = []
    
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            Watcher.count += 1
            print("Yet another class to be finished: " + name)
        Watcher.instances.append(cls)
        super(Watcher, cls).__init__(name, bases, clsdict)


class ToDo(metaclass=Watcher):
    """
    This class will print that something is to be done
    """
    
    @staticmethod
    def status():
        print("Classes to be finished: {}".format(Watcher.count))

    @staticmethod
    def instances():
        return Watcher.instances[:]


class LocalRepoClass:
    """
    Class for local repository object, representing proper configuration space for the library.
    LocalRepoClass generates LocalRepo singleton, which is used as a persistence mechanism for the library
    """

    def __init__(self):
        super().__setattr__("_config", {})

    def __getattr__(self, k):
        if hasattr(self, k):
            return super().__getattribute__(k)
        return self._config[k]

    def __setattr__(self, k, v):
        if hasattr(self, k):
            super().__setattr__(k, v)
            return
        self._config[k] = v

    @staticmethod
    def default():
        pass


LocalRepo = LocalRepoClass.default()
