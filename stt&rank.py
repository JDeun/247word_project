import openai
import os
import nltk
import requests
from kiwipiepy import Kiwi
from collections import Counter
import logging
from dotenv import load_dotenv
import re
import unicodedata
import hanja

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# nltk 데이터 다운로드
nltk.download('punkt', quiet=True)

def transcribe_audio(file_path):
    """음원 파일을 STT 처리하여 텍스트로 변환"""
    try:
        with open(file_path, 'rb') as audio_file:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers={'Authorization': f'Bearer {openai.api_key}'},
                files={'file': audio_file},
                data={'model': 'whisper-1'}
            )
        response.raise_for_status()
        return response.json()['text']
    except FileNotFoundError:
        logging.error(f"The audio file at {file_path} was not found.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while making the API request: {e}")
        return None
    except KeyError:
        logging.error("The API response did not contain the expected 'text' key.")
        return None

def load_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return set()

# 비속어, 한국어 stopwords, 흔한 이름 로드
load_profanities = lambda file_path: load_file(file_path)
load_korean_stopwords = lambda file_path: load_file(file_path)
load_common_names = lambda file_path: load_file(file_path)

def normalize_korean(text):
    # 한글 자모 정규화
    text = unicodedata.normalize('NFC', text)
    
    # 한자를 한글로 변환
    text = hanja.translate(text, 'substitution')
    
    # 영어 소문자화
    text = text.lower()
    
    # 불필요한 공백 제거
    text = ' '.join(text.split())
    
    return text

def is_valid_word(word, pos):
    # 유효한 품사 목록
    valid_pos = ['NNG', 'NNP', 'VV', 'VA']
    # 최소 단어 길이
    min_length = 2
    
    return pos.startswith(tuple(valid_pos)) and len(word) >= min_length and not word.isdigit()

def split_sentences(text):
    # 문장 종결 표현을 기준으로 문장 분리
    sentence_enders = r'(?<=[.!?])\s+(?=[가-힣A-Za-z])|(?<=[.!?])$|(?<=요)\s+(?=[가-힣A-Za-z])|(?<=야)\s+(?=[가-힣A-Za-z])'
    sentences = re.split(sentence_enders, text)
    
    # 추가적인 분리 로직
    refined_sentences = []
    for sentence in sentences:
        # 대화체에서 자주 사용되는 표현을 기준으로 추가 분리
        sub_sentences = re.split(r'(?<=\s)(?:그래서|그런데|그리고|하지만|근데)\s+', sentence)
        refined_sentences.extend(sub_sentences)
    
    return [s.strip() for s in refined_sentences if s.strip() and len(s.strip()) > 5]

def filter_text(text, profanities, korean_stopwords, common_names):
    kiwi = Kiwi()
    
    # 문장 분리
    sentences = split_sentences(text)
    
    filtered_sentences = []
    filtered_words = []
    
    for sentence in sentences:
        # 비속어 필터링
        if not any(profanity in sentence for profanity in profanities):
            normalized_sentence = normalize_korean(sentence)
            
            # 형태소 분석
            morphs = kiwi.analyze(normalized_sentence)
            
            # 유효한 단어만 선택하고 불용어, 고유명사, 흔한 이름 제거
            valid_words = [token.lemma for token in morphs[0][0] 
                           if is_valid_word(token.form, token.tag) 
                           and token.lemma not in korean_stopwords 
                           and token.lemma not in common_names
                           and len(token.lemma) > 1]  # 1글자 단어 제외
            
            # 유효한 단어가 3개 이상인 경우에만 문장 추가
            if len(valid_words) >= 3:
                filtered_sentences.append(normalized_sentence)
                filtered_words.extend(valid_words)

    return filtered_sentences, filtered_words

def get_top_items(items, num_items):
    counter = Counter(items)
    return counter.most_common(num_items)

def main(audio_file_path, profanities_file_path, korean_stopwords_file_path, common_names_file_path, num_sentences, num_words):
    text = transcribe_audio(audio_file_path)
    if text is None:
        return

    profanities = load_profanities(profanities_file_path)
    korean_stopwords = load_korean_stopwords(korean_stopwords_file_path)
    common_names = load_common_names(common_names_file_path)

    filtered_sentences, filtered_words = filter_text(text, profanities, korean_stopwords, common_names)

    top_sentences = get_top_items(filtered_sentences, num_sentences)
    top_words = get_top_items(filtered_words, num_words)

    logging.info(f"상위 문장 {num_sentences}개:")
    for sentence, count in top_sentences:
        logging.info(f"- {sentence} (빈도: {count})")

    logging.info(f"\n상위 단어 {num_words}개:")
    for word, count in top_words:
        logging.info(f"- {word} (빈도: {count})")

if __name__ == "__main__":
    audio_file_path = "C:/Users/gadi2/Downloads/1.mp3"
    profanities_file_path = 'fword_list.txt'
    korean_stopwords_file_path = 'korean_stopwords.txt'
    common_names_file_path = 'common_names.txt'
    
    try:
        num_sentences = int(input("상위 문장 몇 개를 출력하시겠습니까? "))
        num_words = int(input("상위 단어 몇 개를 출력하시겠습니까? "))
    except ValueError:
        logging.error("숫자를 입력해주세요.")
        exit(1)

    main(audio_file_path, profanities_file_path, korean_stopwords_file_path, common_names_file_path, num_sentences, num_words)