import pkg_resources
import subprocess

# pkg_resources.resource_string

class Docker:
    """
    Makes Dockerfiles based on resources
    """

    def __init__(self, resoure_string):
        self.path = pkg_resources.resource_string(__name__, os.path.join("etc", "Dockerfile.template"))

    def create(self):
        p = subprocess.Popen(["docker", ""])