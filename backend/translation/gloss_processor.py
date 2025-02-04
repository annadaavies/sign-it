from .queue import Queue
from .dictionary import Dictionary

class GlossProcessor: 
    def __init__(self, dictionary): 
        self.dictionary = dictionary
        self.punctuation = Dictionary()
        for punctuation in [',','.','?','!']: 
            self.punctuation.add(punctuation, True)
        self.max_word_length = 20
        
    def create_processing_queue(self, structured_gloss): 
        queue = Queue()
        
        for item in structured_gloss: 
            
            if self.is_punctuation(item): 
                queue.enqueue(('pause', 0.5))
            
            elif self.is_compound_word(item): 
                self.process_compound(item, queue)
                
            elif self.dictionary.get(item): 
                queue.enqueue(('word', item))
                
            else: 
                self.process_letters(item, queue)
                
        return queue
    
    def is_punctuation(self, item): 
        for char in item: 
            if char in self.punctuation: 
                return True
        return False
    
    def is_compound_word(self, item): 
        if '-' in item and len(item) <= self.max_word_length: 
            return True
        return False
    
    def process_compound(self, compound_word, queue): 
        components = compound_word.split('-')
        for component in components: 
            if self.dictionary.get(component): 
                queue.enqueue(('word', component))
            else: 
                self.process_letters(component, queue)
                
        queue.enqueue(('compound-end', compound_word))
    
    
    def process_letters(self, word, queue): 
        for char in word: 
            if char.isalpha(): 
                queue.enqueue(('letter', char))
        queue.enqueue(('word-end', word))

