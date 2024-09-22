import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API 키
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # 파일 경로
    PROFANITIES_FILE = 'fword_list.txt'
    KOREAN_STOPWORDS_FILE = 'korean_stopwords.txt'
    COMMON_NAMES_FILE = 'common_names.txt'
    PROMPT_FILE = 'advanced_prompt.txt'

    # OpenAI 모델 설정
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    TOP_P = 1.0
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0
    STOP_SEQUENCES = None

    # 기타 설정
    NUM_SENTENCES = 5
    NUM_WORDS = 20