from .gloss_parser import GlossParser
from .dictionary import Dictionary


class Translator: 
    
    def __init__(self): 
        self.dictionary = Dictionary()
        self.dictionary.load_from_file('backend/translation/mappings.txt')
        self.parser = GlossParser()
        
        def translate_sentence(self, text): 
            processing_queue = self.parser.parse(text) 
            translation_sequence = self.process_queue(processing_queue)
            
            return translation_sequence
        
        def process_queue(self, queue): 
            sequence = []
            while not queue.is_empty(): 
                item_type, value = queue.dequeue()
                
                if item_type == 'word': 
                    sequence.append(self.create_word_entry(value))
                elif item_type == 'letter':
                    sequence.append(self.create_letter_entry(value))
                elif item_type == 'compound-end': 
                    sequence.append(self.create_compound_marker(value))
                elif item_type == 'pause': 
                    sequence.append({'type': 'pause', 'duration': value})
                    
                return sequence