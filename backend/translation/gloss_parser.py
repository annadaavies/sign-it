#NOTE: When documenting, make sure to highlight composition relationship here. 
from .queue import Queue
from .gloss_grammar import GlossGrammar
from .gloss_processor import GlossProcessor
from .dictionary import Dictionary #Only imported for __init__ type hint. 

class GlossParser: 
    """
    Combines grammatical parsing and image processing queue generation. Essentially orchestrates the converting of English text to image processing queue.
    """
    def __init__(self, dictionary: Dictionary): 
        """
        Initialise parser with required components (grammar parser and image queue generator). 
        
        Args: 
        dictionary: Custom dictionary type for word lookups. 
        """
        self.grammar_parser = GlossGrammar()
        self.gloss_processor = GlossProcessor(dictionary) 
        
    def parse(self, text: str) -> Queue: 
        """
        Parse English text into an image processing queue. 
        
        Args: 
        text: English text inputted by the user into the interface.
        
        Returns: 
        Queue: Image processing queue for translation (to be displayed to user through interface).
        """
        structured_gloss = self.grammar_parser.prase_grammar(text) 
        
        processing_queue = self.gloss_processor.create_processing_queue(structured_gloss) 
        
        return processing_queue


    
    