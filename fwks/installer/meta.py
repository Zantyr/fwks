class Dependency(type):
    """
    Metaclass that registers and manages dependencies.

    Each dependency should be created with this metaclass.
    """

    _instances = {}

    def __new__(cls, name, bases, dct):
        if any([k not in dict.keys() for k in ["installer", "is_installed"]]):
            raise RuntimeError("Registered dependency does not have basic handlers")
        item = super().__new__(name, bases, dct)
        cls._instances[name.lower()] = item
        return item

    @classmethod
    def all_instances(cls):
        return cls._instances
