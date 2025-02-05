from .dictionary import Dictionary 

class SignEntry: 
    @staticmethod 
    def create(entry_type, **properties): 
        entry = Dictionary()
        entry.add('type', entry_type)
        for key, value in properties.items(): 
            if value is not None: 
                entry.add(key, value) 
        
        return entry 
    
    @staticmethod
    def to_api_response(entry): 
        response = Dictionary() 
        current = entry.table[0]
        while current: 
            response.add(current.key, current.value) 
            current = current.next
        return response
    
    