from .queue import Queue
from .dictionary import Dictionary

class GlossProcessor: 
    """
    Converts structured ASL gloss into an image processing queue. 
    
    Handles word decomposition (into letters if a direct word translation isn't available), punctuation, and compound words. 
    """
    def __init__(self, dictionary: Dictionary): 
        """
        Initialise processor with dictionary and punctuation rules. 
        
        Args: 
        dictionary: Custom dictionary for word lookups.
        """
        self.dictionary = dictionary
        self.punctuation = Dictionary()
        for punctuation in [',','.','?','!']: 
            self.punctuation.add(punctuation, True)
        self.max_word_length = 20
        
    def create_processing_queue(self, structured_gloss: list) -> Queue: 
        """
        Create image processing queue from structured ASL gloss. 
       
        Args: 
        structured_gloss (list): Grammatically structured ASL gloss words. 
       
        Returns: 
        Queue: Image processing queue for display to interface. 
        """
        
        queue = Queue()
        
        for item in structured_gloss: 
            
            if self.is_punctuation(item): #If the item is determined to be a punctuation mark. 
                queue.enqueue(('pause', 0.5))
            
            elif self.is_compound_word(item): #If the item is determined to be a compound word.
                self.process_compound(item, queue)
                
            elif self.dictionary.get(item.lower()): #If the item exists in the word dictionary (from the file). 
                queue.enqueue(('word', item.lower()))
                
            else: 
                self.process_letters(item, queue)
                
        return queue
    
    def is_punctuation(self, item: str) -> bool: 
        """
        Check if word contains punctuation.
        
        Args: 
        item: Input string to check.
        
        Returns: 
        bool: Boolean flag to indicate whether punctuation is found. True if punctuation in word, False otherwise. 
        """
        for char in item: 
            if char in self.punctuation: 
                return True
        return False
    
    def is_compound_word(self, item: str) -> bool: 
        """
        Check if item is a compound word. 
        
        Args:
        item: Input string to check. 
        
        Returns: 
        bool: Boolean flag to indicate whether the item is a compound word. True if it is, False otherwise. 
        """
        if '-' in item and len(item) <= self.max_word_length: 
            return True
        return False
    
    def process_compound(self, compound_word: str, queue: Queue): 
        """
        Process compound word into its components. 
        
        Args: 
        compound_word: Compound word to process. 
        queue: Processing queue to add components to. 
        """
        components = compound_word.split('-')
        for component in components: 
            if self.dictionary.get(component.lower()): #If part of the compound word exists in the defined word dictionary, add.
                queue.enqueue(('word', component.lower()))
            else: 
                self.process_letters(component, queue) #If not, process the word into letters.
                
        queue.enqueue(('compound-end', compound_word)) 
    
    
    def process_letters(self, word: str, queue: Queue): 
        """
        Process word into indivdual letters. 
        
        Args: 
        word: Word to process. 
        queue: Processing queue to add letters to. 
        """
        for char in word: 
            if char.isalpha(): 
                queue.enqueue(('letter', char))
        queue.enqueue(('word-end', word))

