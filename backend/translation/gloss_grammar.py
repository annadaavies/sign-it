from .dictionary import Dictionary

class GlossGrammar: 
    
    def __init__(self): 
        self.stop_words = Dictionary()
        for word in ['a', 'and', 'the', 'and', 'but', 'or']:
            self.stop_words.add(word, True) 
        
        self.asl_structures = Dictionary()
        self.asl_structures.add('time-first', ['tomorrow', 'yesterday', 'today', 'next-week'])
        self.asl_structures.add('topic-comment', ['who', 'what', 'where', 'why', 'how'])

        
    def parse_grammar(self,text): 
        words = self.preprocess(text) 
        structured_words = self.apply_grammar_rules(words) 
        
        return self.finalise_structure(structured_words)
    
    def preprocess(self, text): 
        processed_words_list = [word.lower() for word in text.split() if word.lower() not in self.stop_words]
        return processed_words_list
            
    def apply_grammar_rules(self, words): 
        time_words = [word for word in words if word in self.asl_structures.get('time-first')]
        
        if time_words: 
            return time_words + [word for word in words if word not in time_words]
        
        question_words = [word for word in words if word in self.asl_structures.get('topic-comment')]
        
        if question_words: 
            return question_words + [word for word in words if word not in question_words]
        
        if len(words) >= 3: 
            return [words[2], words[0], words[1]] + words[3:]
        return words
    
    def finalise_structure(self, words):
        structured_words = [word.upper() if '-' not in word else word for word in words]
        return structured_words
        
        
        
        
    