import re
import math
import unicodedata
from itertools import product
import os

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Empirically derived mean sentiment intensity rating increase for booster words
B_INCR = 0.293
B_DECR = -0.293

# For removing punctuation
REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

# Check for special case idioms containing lexicon words
SPECIAL_CASE_IDIOMS = {
    "top": 2, 
    "daora": 1, 
    "massa": 1, 
    "zoado": -2, 
    "mó ruim": -2,
    "de boa": 1, 
    "tenso": -1,
    "imbatível": 2.5
}

ADVERSATIVES = [
    "mas", 
    "porem", 
    "contudo", 
    "entretanto",
    "todavia"
]

class SentimentIntensityAnalyzer(object):
    def __init__(
        self,
        lexicon_file=os.path.join(
            PACKAGE_DIRECTORY,
            'lexicons',
            'vader_lexicon_ptbr.txt'
        ),
        emoji_lexicon=os.path.join(
            PACKAGE_DIRECTORY,
            'lexicons',
            'emoji_utf8_lexicon_ptbr.txt'
        ),
        booster_lexicon=os.path.join(
            PACKAGE_DIRECTORY, 
            'lexicons', 
            'booster.txt'
        ),
        negate_lexicon=os.path.join(
            PACKAGE_DIRECTORY, 
            'lexicons', 
            'negate.txt'
        )
    ):
        with open(lexicon_file, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        with open(emoji_lexicon, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

        with open(booster_lexicon, encoding='utf-8') as f:
            self.booster_full_filepath = f.read()
        self.booster = self.make_booster_dict()

        with open(negate_lexicon, encoding='utf-8') as f:
            self.negate = [t.strip() for t in f]



    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.split('\n'):
            if len(line) < 1:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict


    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.split('\n'):
            if len(line) < 1:
                continue
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict
    
    def make_booster_dict(self):
        """
        Convert booster lexicon file to a dictionary
        """
        boosters = []
        for boost in self.booster_full_filepath.split('\n'):
            parts = boost.strip().split(' ')
            boosters.append([' '.join(parts[:-1]), parts[-1]])

        booster_dict = {}
        for t, v in boosters: 
            booster_dict[t] = B_INCR if v == 'INCR' else B_DECR
        
        return booster_dict

    def polarity_scores(self, text):
        # Remove acentos
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

        #Remove pontuacao
        text = REGEX_REMOVE_PUNCTUATION.sub('', text)
        
        words = text.split()
        score = 0

        i = 0
        boost = [0]
        negating = [False]
        valence = 0

        while i < len(words):
            
            item = words[i]
            item_lowercase = item.lower()

            bigrama = f"{item_lowercase} {words[i+1]}" if i+1 < len(words) else None

            #Calcular quantidade de boost
            if item_lowercase in self.booster:
                boost[0] += self.booster[item_lowercase]
                i+=1
                continue
                
            if item_lowercase in ADVERSATIVES:
                score = 0
                if boost[0] != 0:
                    if valence < 0:
                        boost[0] *= -1
                    score += boost[0]

                i+=1
                continue
            
            if item_lowercase in self.negate:
                negating[0] = True
                i+=1
                continue
            
            if item_lowercase in self.lexicon:
                # Get the sentiment valence
                valence = self.calculate_sentiment(item_lowercase, self.lexicon, boost, negating)
                score += valence

            elif bigrama and bigrama in self.lexicon:
                # Get the sentiment valence
                valence = self.calculate_sentiment(bigrama, self.lexicon, boost, negating)
                score += valence
            
            elif item_lowercase in SPECIAL_CASE_IDIOMS:
                # Get the sentiment valence
                valence = self.calculate_sentiment(item_lowercase, SPECIAL_CASE_IDIOMS, boost, negating)
                score += valence
            
            elif bigrama and bigrama in SPECIAL_CASE_IDIOMS:
                # Get the sentiment valence
                valence = self.calculate_sentiment(bigrama, SPECIAL_CASE_IDIOMS, boost, negating)
                score += valence

            i+=1
        
        if boost[0] != 0:
            if valence < 0:
                boost[0] *= -1
            score += boost[0]

        return score


    def calculate_sentiment(self, item, lexicon_dict, boost, negating):
        valence = lexicon_dict[item]
        
        if valence < 0:
            boost[0] *= -1
                
        valence += boost[0]
        boost[0] = 0

        if negating[0] == True:
            valence *= -1
            negating[0] = False
        
        return valence

if __name__ == '__main__':
    pass
