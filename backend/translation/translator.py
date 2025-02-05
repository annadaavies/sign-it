from .gloss_parser import GlossParser
from .dictionary import Dictionary
from .sign_entry import SignEntry


class Translator: 
    
    def __init__(self): 
        self.dictionary = Dictionary()
        self.dictionary.load_from_file('backend/translation/mappings.txt')
        self.parser = GlossParser(self.dictionary)
        
        def translate_sentence(self, text): 
            processing_queue = self.parser.parse(text) 
            translation_sequence = self.process_queue(processing_queue)
            
            return translation_sequence
        
        def _process_queue(self, queue): 
            sequence = Dictionary() 
            index = 0
            while not queue.is_empty(): 
                item_type, value = queue.dequeue()
                entry = self.create_entry(item_type, value) 
                sequence.add(str(index), entry)
                index += 1
            return sequence
        
        def _create_entry(self, item_type, value): 
            if item_type == 'word': 
                return SignEntry.create('video', label=value.upper(), value=self.dictionary.get(value))
            
            elif item_type == 'letter': 
                return SignEntry.create('image', label=value.upper(), value=f"{value.lower()}.jpg", duration=2.0)
            
            elif item_type == 'pause': 
                return SignEntry.create('pause', duration=value) 
            
            elif item_type == 'compound-end': 
                return SignEntry.create('animation', label=value.upper(), duration=1.0)
            
            return Dictionary()
    
            