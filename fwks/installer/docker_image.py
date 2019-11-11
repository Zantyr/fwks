from fwks.installer.meta import Dependency



class DockerImage(metaclass=Dependency):
    @classmethod
    def is_installed(self):
        pass

    @classmethod
    def installer(self):
        pass

"""
pynini
sox
sox-format-handlers
carfac
docker-image
"""
