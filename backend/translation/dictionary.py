MAX_DICTIONARY_SIZE = 1000

class DictionaryEntry: 
    """
    A entry representing a key-value pair in the dictionary. 
    """
    def __init__(self, key: str, value: any): 
        """
        Initialise an empty dictionary entry. 
        
        Args: 
        key: Unique identifier for the entry. 
        value: Corresponding data to store with the key. 
        """
        self.key = key
        self.value = value
        self.next = None #Each entry stores reference to the next entry in hash table. 

class Dictionary: 
    """
    Hash table implementation with collision handling. 
    """
    def __init__(self, size: int = MAX_DICTIONARY_SIZE):
        """
        Initialise empty dictionary with specified maximum capacity. 
        
        Args: 
        size: Maximum number of 'places' (each place being a spot to store a key-value entry). 
        """ 
        self.size = size 
        self.table = [None] * size #[None, None, None....] - Each None represents a place for a key-value entry. 
        self.count = 0 #Actual number of entries in the dictionary at any point. 
        
    def hash(self, key: str) -> int: 
        """
        Use simple hash algorithm to compute the index that a entry should be inserted based on a given key. 
        
        Hash algorithm is an ordinal sum hash; hash equals the remainder of the sum of the character codes of each character in the key divided by the maximum size of the dictionary. 
        
        Args: 
        key: Key of the dictionary entry to hash. 
        
        Returns: 
        int: An index indicating the position the key-value entry should be inserted.
        """
        hash_value = sum(ord(char) for char in key) % self.size
        return hash_value
    
    def add(self, key: str, value: any): 
        """
        Insert or update a key-value pair in the dictionary. 
        
        Handles collisions using chaining. 
        
        Args: 
        key: Lookup identifier
        value: 
        
        key: Unique identifier for the entry to add.
        value: Corresponding data to store with the key. 
        """
        if self.count == self.size: 
            raise IndexError("Error: Dictionary is full.")
        
        index = self.hash(key) 
        new_entry = DictionaryEntry(key, value) 
        
        if self.table[index] is None: #Simplest case: the hash index is empty and key-value pair can be inserted. 
            self.table[index] = new_entry
        
        else: 
            current = self.table[index] 
            while current.next: #While the next place in the dictionary is empty. 
                if current.key ==  key: #If a key-value pair with the exact same key already exists in the dictionary, it will just update the value of that key-value pair. 
                    current.value = value
                    return 
                current = current.next
            current.next = new_entry
        self.count += 1
            
    def get(self, key: str) -> any: 
        """
        Retrieve value associated with a key. 
        
        Args: 
        key: Unique identifier for the entry to look up. 
        
        Returns: 
        any: Value associated with the argument key (None if key-value pair not in dictionary). 
        """
        index = self.hash(key)
        current = self.table[index]
        
        while current: 
            if current.key == key: 
                return current.value
            current = current.next #Iterate through each key-value pair in dictionary, checking for a key match. 
        
        return None
    
    def remove(self, key): 
        #TODO: Remember to finish this docstring!!!!!!!
        index = self.hash(key) 
        current = self.table[index]
        
        previous = None
        while current: 
            if current.key == key: 
                if previous: 
                    previous.next = current.next
                else: 
                    self.table[index] = current.next
                self.count -= 1
                
            previous = current
            current = current.next
        return False
    
    def in_dict(self, key: str) -> bool: 
        """
        Check if a key-value pair exists in the dictionary based on the key. 
        
        Args: 
        key: Unique identifier to check existence. 
        
        Returns: 
        bool: Flag to mark whether the key-value pair exists in the dictionary (True if key-value pair in dictionary, False if not).
        """
        key_in_dict = self.get(key) is not None
        return key_in_dict
    
    def load_from_file(self, filename: str): 
        """
        Populate dictionary from a file. 
        
        File must be structured as: 
        key1, value1
        key2, value2
        key3, value3
        ....
        
        Args: 
        filename: Path to input file
        """
        try: 
            with open(filename, 'r') as file: 
                for line_number, line in enumerate(file, 1): 
                    line = line.strip()
                    if not line: 
                        continue
                    
                    if ',' not in line: 
                        raise ValueError("Error: File incorrectly formatted. See docstring for appropriate input file format.")
                    
                    key, value = line.split(',')
                    self.add(key.lower(), value)
                    
        except: 
            raise FileNotFoundError("Error: File not found.")


    def serialise(self): 
        
        result = []
        for entry in self.table: 
            current = entry 
            while current: 
                if isinstance(current.value, Dictionary): 
                    result.append(current.value.serialise())
                else: 
                    result.append({
                        'key': current.key,
                        'value': current.value
                    })
                current = current.nexts
        return result
        
