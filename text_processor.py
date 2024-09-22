import re
import unicodedata
import hanja
from kiwipiepy import Kiwi
from collections import Counter
from config import Config

class TextProcessor:
    def __init__(self):
        self.kiwi = Kiwi()
        self.profanities = self.load_file(Config.PROFANITIES_FILE)
        self.korean_stopwords = self.load_file(Config.KOREAN_STOPWORDS_FILE)
        self.common_names = self.load_file(Config.COMMON_NAMES_FILE)

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def normalize_korean(self, text):
        text = unicodedata.normalize('NFC', text)
        text = hanja.translate(text, 'substitution')
        text = text.lower()
        return ' '.join(text.split())

    def is_valid_word(self, word, pos):
        valid_pos = ['NNG', 'NNP', 'VV', 'VA']
        return pos.startswith(tuple(valid_pos)) and len(word) >= 2 and not word.isdigit()

    def split_sentences(self, text):
        sentence_enders = r'(?<=[.!?])\s+(?=[가-힣A-Za-z])|(?<=[.!?])$|(?<=요)\s+(?=[가-힣A-Za-z])|(?<=야)\s+(?=[가-힣A-Za-z])'
        sentences = re.split(sentence_enders, text)
        refined_sentences = []
        for sentence in sentences:
            sub_sentences = re.split(r'(?<=\s)(?:그래서|그런데|그리고|하지만|근데)\s+', sentence)
            refined_sentences.extend(sub_sentences)
        return [s.strip() for s in refined_sentences if s.strip() and len(s.strip()) > 5]

    def filter_text(self, text):
        sentences = self.split_sentences(text)
        filtered_sentences = []
        filtered_words = []
        
        for sentence in sentences:
            if not any(profanity in sentence for profanity in self.profanities):
                normalized_sentence = self.normalize_korean(sentence)
                morphs = self.kiwi.analyze(normalized_sentence)
                valid_words = [token.lemma for token in morphs[0][0] 
                               if self.is_valid_word(token.form, token.tag) 
                               and token.lemma not in self.korean_stopwords 
                               and token.lemma not in self.common_names
                               and len(token.lemma) > 1]
                if len(valid_words) >= 3:
                    filtered_sentences.append(normalized_sentence)
                    filtered_words.extend(valid_words)

        return filtered_sentences, filtered_words

    def get_top_items(self, items, num_items):
        counter = Counter(items)
        return counter.most_common(num_items)