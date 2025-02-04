#NOTE: When documenting, make sure to highlight composition relationship here. 

from .gloss_grammar import GlossGrammar
from .gloss_processor import GlossProcessor

class GlossParser: 
    
    def __init__(self, dictionary): 
        self.grammar_parser = GlossGrammar()
        self.gloss_processor = GlossProcessor(dictionary) 
        
    def parse(self, text): 
        structured_gloss = self.gloss_grammar.prase_grammar(text) 
        
        processing_queue = self.gloss_process.create_processing_queue(structured_gloss) 
        
        return processing_queue


    
    