
MAX_DICTIONARY_SIZE = 1000

class Dictionary: 

    def __init__(self, size = MAX_DICTIONARY_SIZE): 
        self.size = size
        self.storage = [[] for i in range(self.size)]
        self.length = 0 

    def add(self, key, value): 
        index = hash()




