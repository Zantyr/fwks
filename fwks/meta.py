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
