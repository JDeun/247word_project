import re
import unicodedata
import hanja
from kiwipiepy import Kiwi
from collections import Counter
from config import Config

class TextProcessor:
    """
    텍스트 처리를 위한 클래스.
    이 클래스는 한국어 텍스트를 정제하고, 문장을 분리하며, 유효한 단어를 추출합니다.
    """

    def __init__(self):
        """
        TextProcessor 초기화
        Kiwi 형태소 분석기를 초기화하고, 필요한 단어 목록들을 로드합니다.
        """
        self.kiwi = Kiwi()  # 한국어 형태소 분석을 위한 Kiwi 초기화
        # 비속어, 불용어, 일반적인 이름 목록을 파일에서 로드
        self.profanities = self.load_file(Config.PROFANITIES_FILE)
        self.korean_stopwords = self.load_file(Config.KOREAN_STOPWORDS_FILE)
        self.common_names = self.load_file(Config.COMMON_NAMES_FILE)

    def load_file(self, file_path):
        """
        파일을 읽어 각 줄을 집합(set)으로 반환하는 메서드
        :param file_path: 읽을 파일의 경로
        :return: 파일 내용을 줄 단위로 저장한 집합
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def normalize_korean(self, text):
        """
        한국어 텍스트를 정규화하는 메서드
        :param text: 정규화할 텍스트
        :return: 정규화된 텍스트
        """
        text = unicodedata.normalize('NFC', text)  # 유니코드 정규화
        text = hanja.translate(text, 'substitution')  # 한자를 한글로 변환
        text = text.lower()  # 소문자화
        return ' '.join(text.split())  # 연속된 공백 제거

    def is_valid_word(self, word, pos):
        """
        단어가 유효한지 확인하는 메서드
        :param word: 확인할 단어
        :param pos: 품사 태그
        :return: 유효 여부 (불리언)
        """
        valid_pos = ['NNG', 'NNP', 'VV', 'VA']  # 유효한 품사 목록 (명사, 동사, 형용사)
        return pos.startswith(tuple(valid_pos)) and len(word) >= 2 and not word.isdigit()

    def split_sentences(self, text):
        """
        텍스트를 문장 단위로 분리하는 메서드
        :param text: 분리할 텍스트
        :return: 분리된 문장 리스트
        """
        # 문장 종결 표현을 기준으로 1차 분리
        sentence_enders = r'(?<=[.!?])\s+(?=[가-힣A-Za-z])|(?<=[.!?])$|(?<=요)\s+(?=[가-힣A-Za-z])|(?<=야)\s+(?=[가-힣A-Za-z])'
        sentences = re.split(sentence_enders, text)
        
        refined_sentences = []
        for sentence in sentences:
            # 접속사를 기준으로 2차 분리
            sub_sentences = re.split(r'(?<=\s)(?:그래서|그런데|그리고|하지만|근데)\s+', sentence)
            refined_sentences.extend(sub_sentences)
        
        # 빈 문장 제거 및 최소 길이(5자) 이상인 문장만 선택
        return [s.strip() for s in refined_sentences if s.strip() and len(s.strip()) > 5]

    def filter_text(self, text):
        """
        텍스트를 필터링하여 유효한 문장과 단어를 추출하는 메서드
        :param text: 필터링할 텍스트
        :return: 필터링된 문장 리스트와 단어 리스트의 튜플
        """
        sentences = self.split_sentences(text)
        filtered_sentences = []
        filtered_words = []
        
        for sentence in sentences:
            # 비속어 필터링
            if not any(profanity in sentence for profanity in self.profanities):
                normalized_sentence = self.normalize_korean(sentence)
                morphs = self.kiwi.analyze(normalized_sentence)
                # 유효한 단어 추출
                valid_words = [token.lemma for token in morphs[0][0] 
                               if self.is_valid_word(token.form, token.tag) 
                               and token.lemma not in self.korean_stopwords 
                               and token.lemma not in self.common_names
                               and len(token.lemma) > 1]
                # 유효한 단어가 3개 이상인 문장만 선택
                if len(valid_words) >= 3:
                    filtered_sentences.append(normalized_sentence)
                    filtered_words.extend(valid_words)

        return filtered_sentences, filtered_words

    def get_top_items(self, items, num_items):
        """
        주어진 아이템 리스트에서 가장 빈도가 높은 아이템을 반환하는 메서드
        :param items: 아이템 리스트
        :param num_items: 반환할 상위 아이템 수
        :return: (아이템, 빈도) 튜플의 리스트
        """
        counter = Counter(items)
        return counter.most_common(num_items)