from .dictionary import Dictionary 

class SignEntry: 
    """
    Create sign entries to be added to queue. 
    
    Handles creation and formatting of translation entries. 
    """
    @staticmethod 
    def create(entry_type: str, **properties: Dictionary) -> Dictionary: 
        """
        Create a new sign entry. 
        
        Args: 
        entry_type: Type of entry (video, image, pause, etc.).
        properties: Additional properties for the entry (e.g. duration). 
        
        Returns: 
        Dictionary: Created sign entry. 
        """
        entry = Dictionary()
        entry.add('type', entry_type)
        
        for key, value in properties.items(): 
            if value is not None: 
                entry.add(key, value) 
        
        return entry 
    
    @staticmethod
    def to_api_response(entry: Dictionary) -> Dictionary: 
        """
        Serialise dictionary entry to native Python types (such that they can be 'jsonified'). 
        
        Args: 
        entry: Entry to serialise. 
        
        Returns: 
        Dictionary: Serialised entry. 
        """
        response = Dictionary() 
        current = entry.table[0]
        while current: 
            response.add(current.key, current.value) 
            current = current.next
        return response
    
    