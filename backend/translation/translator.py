class Translator: 
    
    def __init__(self): 
        self.word_mappings = self.load_mappings()
        
        def load_mappings(self): 
            #Load from file. Use custom built dictionary class. 
            pass

        def translate_word(self, word): 
            #get here is a dictionary built-in function, needs to be implemented with own dictionary class. 
            return self.word_mappings.get(word.lower(), None)
        
        def translate_sentence(self, gloss_sentence): 
            #gloss_sentence should already be a list of words in correct order. 
            #NOTE: Should also add some sort of space sign in between words. 
            #NOTE: All uses of dictionaries need to be replaced with dicionary built from scratch. 
            #NOTE: Could replace for letter in word with some sort of queue process? 
            translated_sequence = []
            
            for word in gloss_sentence: 
                translation = self.translate_word(word)
                if translation: 
                    translated_sequence.append({
                        'type': 'video' if translation.endswith('.mp4') else 'image', 
                        'value': translation, 
                        'label': word
                    })
                else: 
                    for letter in word: 
                        translated_sequence.append({
                            'type': 'image',
                            'value': f'{letter.lower()}.jpg',
                            'label': letter
                        })
            
            return translated_sequence