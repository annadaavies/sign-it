from .gloss_parser import GlossParser
from .dictionary import Dictionary
from .sign_entry import SignEntry
from .queue import Queue


class Translator: 
    """
    Main translation class handling the conversion from English to ASL signs. 
    
    Uses custom data structures throughout translation process. 
    """
    
    def __init__(self): 
        """
        Initialise translator with dictionary and parser.
        """
        self.dictionary = Dictionary()
        self.dictionary.load_from_file('backend/translation/mappings.txt')
        self.parser = GlossParser(self.dictionary)
        
    def translate_text(self, text: str) -> Dictionary: 
        """
        Translate English sentence to ASl signs.
            
        Args: 
        text: Input English text
            
        Returns: 
        list: Dictionary of sign entries (custom dictionary types) representing ASL signs. 
        """
        processing_queue = self.parser.parse(text) #Transform English text to ASL gloss.
            
        translation_sequence = self._process_queue(processing_queue) #Transform ASL gloss to image processing queue. 
            
        return translation_sequence
        
    def _process_queue(self, queue: Queue) -> Dictionary: 
        """
        Process translation queue into sign sequence.  
            
        Args: 
        queue: Processing queue 
            
        Returns: 
        Dictionary: Dictionary sequence of sign entries. 
        """
        sequence = Dictionary() 
        index = 0
            
        while not queue.is_empty(): 
            item_type, value = queue.dequeue()
            entry = self._create_entry(item_type, value) 
            sequence.add(str(index), entry)
            index += 1
        return sequence
        
    def _create_entry(self, item_type: str, value: str) -> Dictionary: 
        """
        TODO: Final function from translator.py to fill in. 
        """
        if item_type == 'word': 
            return SignEntry.create('video', label=value.upper(), value=self.dictionary.get(value))
            
        elif item_type == 'letter': 
            return SignEntry.create('image', label=value.upper(), value=f"{value.lower()}.jpg", duration=2.0)
            
        elif item_type == 'pause': 
            return SignEntry.create('pause', duration=value) 
            
        elif item_type == 'compound-end': 
            return SignEntry.create('animation', label=value.upper(), duration=1.0)
            
        return Dictionary()
    
            