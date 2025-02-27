from .dictionary import Dictionary

class GlossGrammar: 
    """
    Handles conversion of English text (written with English syntax) to the correct ASL grammatical structure (ASL gloss - formatted in CAPITALS). 

    Implementation includes: 
    - Removal of filler/stop words (e.g., 'a', 'and', 'the', etc.).
    - Prioritisation of 'time-first' words in sentences (e.g., 'yesterday', 'today', 'tomorrow').
    - Correct placement of question words (e.g., 'who, 'what', 'where', 'why', 'how').
    - Final conversion to uppercase gloss (unless the word is a compound word, i.e. contains '-').
    """
    def __init__(self): 
        """
        Initialise grammar rules and stop words. 
        """
        self.stop_words = Dictionary() #Stop word dictionary. These are words that are rarely signed in ASL.
        for word in ['a', 'and', 'the', 'and', 'but', 'or']:
            self.stop_words.add(word, True) 
        
        self.asl_structures = Dictionary()
        self.asl_structures.add('time-first', ['tomorrow', 'yesterday', 'today', 'next-week']) #Time-first words, in ASL should appear at the beginning of a sentence when present. 
        self.asl_structures.add('topic-comment', ['who', 'what', 'where', 'why', 'how']) #Topic-comment words, structured in accordance to ASL's topic-comment sentence structure. 

        
    def parse_grammar(self, text: str) -> list: 
        """
        Main method that translates English syntax to ASL syntax (ASL gloss) through the smaller steps of:  
        - Preprocessing (stop word filtering, lowercasing)
        - Applying ASL-specific grammar rules (time-first, topic-comment question words). 
        - Finalising the structure (capitalising to indicate ASL gloss) 
        
        Args: 
        text (str): English text inputted by the user into the interface.
        
        Returns: 
        list: List of structured words in ASL grammatical order (i.e. ASL gloss order). 
        """
        words = self.preprocess(text) 
        structured_words = self.apply_grammar_rules(words) 
        
        return self.finalise_structure(structured_words)
    
    def preprocess(self, text: str) -> list: 
        """
        Format input text in preparation for restructuring and parsing to ASL gloss.
        
        Strip out stop words, convert everything to lowercase. 
        
        Args: 
        text: English text inputted by the user into the interface.
        
        Returns: 
        list: List of processed (as defined above) words. 
        """
        words = text.split()
        processed_words_list = [word.lower() for word in words if word.lower() not in self.stop_words]
        
        return processed_words_list
            
    def apply_grammar_rules(self, words: list) -> list: 
        """
        Apply ASL-specific grammar rules to word sequence. 
        
        - Move 'time-first' words to the front if any exist.
        - Move question words to the end.
        - Keep everything else in between in the same order as they appear.     
        
        Args: 
        words: List of input words (lowercase, excluding stop words). 
        
        Returns: 
        list: List of words reordered according to ASL grammar rules. 
        """
        
        time_words = []
        question_words = []
        middle_words = []
        
        time_list = self.asl_structures.get('time-first') or []
        question_list = self.asl_structures.get('topic-comment') or []
        
        for word in words: 
            if word in time_list: 
                time_words.append(word)
            elif word in question_list: 
                question_words.append(word) 
            else: 
                middle_words.append(word)
        time_words = [word for word in words if word in self.asl_structures.get('time-first')] #Collect list of words in the input words list that are also in the time-first words dictionary. 
        
        return time_words + middle_words + question_words
    
    def finalise_structure(self, words: list) -> list:
        """
        Perform final formatting of ASL gloss structure (ASL gloss, as mentioned in report, is typically formatted in CAPITAL PRINT). 
        
        Args: 
        words: ASL gloss words in list form. 
        
        Returns: 
        list: Final formatted ASL gloss words in list form. 
        
        """
        structured_words = [word.upper() if '-' not in word else word for word in words] #Capitalise all non-compound words. Compound words are typically left lowercase in ASL gloss. 
        
        return structured_words
    
        
        
        
        
    