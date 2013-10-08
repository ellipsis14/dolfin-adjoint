class Parameters(dict):
    def __init__(self, name=None):
        self.name = name

    def add(self, key, value=None):
        if value is not None:
            self[key] = value 
        else:
            self[key.name] = key
            
parameters = Parameters()
