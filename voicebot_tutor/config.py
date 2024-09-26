import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 설정
LLM_MODEL = "gpt-4o-mini"  # 대화 생성을 위한 GPT 모델
TTS_MODEL = "tts-1"  # Text-to-Speech 모델
STT_MODEL = "whisper-1"  # Speech-to-Text 모델

# 음성 설정
VOICE_OPTION = "nova"  # TTS 음성 옵션
TTS_SPEED = 1.2  # TTS 음성 속도 (1.0이 기본 속도)

# 대화 설정
MAX_TOKENS = None  # 생성할 최대 토큰 수
TEMPERATURE = 0.7  # 응답의 다양성 조절 (0.0 ~ 1.0)

# 프롬프트 파일 읽기
with open("prompt.txt", "r", encoding="utf-8") as file:
    SYSTEM_MESSAGE = file.read()

# 초기 인사 메시지
INITIAL_GREETING = "안녕! 오늘은 무슨 일이 있었어? 그것에 대한 영어 표현을 배워보는 건 어때?"