from .dictionary import Dictionary

class GlossGrammar: 
    """
    Handles conversion of English text (written with English syntax) to the correct ASL grammatical structure (ASL gloss - formatted in CAPITALS). 

    Implementation involves consideration of time-first words and ASL's topic-comment structure. 
    """
    def __init__(self): 
        """
        Initialise grammar rules and stop words. 
        """
        self.stop_words = Dictionary() #Stop word dictionary. These are words that are typically not signed as frequently in ASL as in English. 
        for word in ['a', 'and', 'the', 'and', 'but', 'or']:
            self.stop_words.add(word, True) 
        
        self.asl_structures = Dictionary()
        self.asl_structures.add('time-first', ['tomorrow', 'yesterday', 'today', 'next-week'])
        self.asl_structures.add('topic-comment', ['who', 'what', 'where', 'why', 'how'])

        
    def parse_grammar(self, text: str) -> list: 
        """
        Main method that combines other defined methods to convert English text (i.e. English syntax) to ASL gloss. 
        
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
        Format input text in preparation for restructuring (following grammar rules) and parsing. 
        
        Adds the lowercase version of each word in the text that are not stop words. 
        
        Args: 
        text: English text inputted by the user into the interface.
        
        Returns: 
        list: List of processed (as defined above) words. 
        """
        
        processed_words_list = [word.lower() for word in text.split() if word.lower() not in self.stop_words] #Split text into a list, iterate through each word, transform to lowercase version, and add to processed_words_list if not a stop word. 
        return processed_words_list
            
    def apply_grammar_rules(self, words: list) -> list: 
        """
        Apply ASL-specific grammar rules to word sequence. 
        
        Args: 
        words: List of input words (lowercase, excluding stop words). 
        
        Returns: 
        list: List of words reordered according to ASL grammar rules. 
        """
        time_words = [word for word in words if word in self.asl_structures.get('time-first')] #Collect list of words in the input words list that are also in the time-first words dictionary. 
        
        if time_words: 
            return time_words + [word for word in words if word not in time_words] #In ASL grammar, time words take priority in a sentence. 
        
        question_words = [word for word in words if word in self.asl_structures.get('topic-comment')] #Collect list of words 
        
        if question_words: 
            return question_words + [word for word in words if word not in question_words]
        
        if len(words) >= 3: 
            return [words[2], words[0], words[1]] + words[3:]
        return words
    
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
    
        
        
        
        
    