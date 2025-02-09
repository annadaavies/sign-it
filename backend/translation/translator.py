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
        Create sign entry from queue item using a mapping approach. 
        
        Args: 
        item_type: Type of entry to create (word, letter, pause, compound-end). 
        value: Value to use for the entry. 
        
        Returns: 
        Dictionary: Configured sign entry. 
        """
        
        entry = Dictionary() 
        
        if item_type == 'word': 
            entry.add('type', 'video')
            entry.add('label', value.upper())
            entry.add('value', self.dictionary.get(value))
            
        elif item_type == 'letter': 
            entry.add('type', 'image')
            entry.add('label', value.upper())
            entry.add('value', f"{value.lower()}.jpg")
            entry.add('duration', 2.0)
            
        elif item_type == 'pause':
            entry.add('type', 'pause')
            entry.add('duration', 0.5) 
            
        elif item_type == 'compound-end': 
            entry.add('type', 'animation')
            entry.add('label', value.upper())
            entry.add('duration', 1.0)
            
        return entry
    
            