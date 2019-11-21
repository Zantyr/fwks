import luigi

class MyTask(luigi.Task):
    def run(self):
        print("Gello Welt")
