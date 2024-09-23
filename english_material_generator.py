import json
import logging
from openai import OpenAI
from config import Config

class EnglishMaterialGenerator:
    """
    OpenAI GPT 모델을 사용하여 영어 학습 자료를 생성하는 클래스.
    이 클래스는 주어진 문장과 단어를 기반으로 대화와 어휘 목록을 생성합니다.
    """

    def __init__(self):
        """
        EnglishMaterialGenerator 초기화
        OpenAI 클라이언트를 설정하고 프롬프트 파일을 로드합니다.
        """
        # OpenAI API 클라이언트 초기화
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # 프롬프트 파일 로드
        with open(Config.PROMPT_FILE, 'r', encoding='utf-8') as f:
            self.prompt_content = f.read()

    def generate_material(self, sentences, words):
        """
        영어 학습 자료를 생성하는 메서드
        
        :param sentences: 입력 문장 리스트 (각 요소는 (문장, 빈도) 튜플)
        :param words: 입력 단어 리스트 (각 요소는 (단어, 빈도) 튜플)
        :return: 생성된 학습 자료 (딕셔너리 형태) 또는 오류 시 None
        """
        try:
            # 프롬프트 파일에서 시스템 메시지와 사용자 메시지 분리
            system_message, user_message = self.prompt_content.split("[사용자 메시지]")
            system_message = system_message.replace("[시스템 메시지]\n", "").strip()
            user_message = user_message.strip()

            # 사용자 메시지 포맷팅
            formatted_user_message = user_message.format(
                sentences="\n".join(f"- {sentence}" for sentence, _ in sentences),
                words=", ".join(word for word, _ in words)
            )

            # GPT API 호출
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": formatted_user_message}
                ],
                temperature=Config.TEMPERATURE,  # 출력의 무작위성 조절
                max_tokens=Config.MAX_TOKENS,    # 최대 토큰 수 제한
                top_p=Config.TOP_P,              # 상위 확률 샘플링
                frequency_penalty=Config.FREQUENCY_PENALTY,  # 단어 반복 억제
                presence_penalty=Config.PRESENCE_PENALTY,    # 새로운 주제 도입 장려
                stop=Config.STOP_SEQUENCES       # 응답 종료 시퀀스
            )

            # API 응답에서 콘텐츠 추출 및 정제
            content = response.choices[0].message.content
            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```").strip()
            if content.startswith("json"):
                content = content.replace("json", "", 1).strip()

            # JSON 파싱
            return json.loads(content)

        except json.JSONDecodeError as e:
            # JSON 파싱 오류 처리
            logging.error(f"GPT response is not a valid JSON. Error: {str(e)}")
            logging.error(f"Raw response: {content}")
        except Exception as e:
            # 기타 예외 처리
            logging.error(f"An unexpected error occurred: {str(e)}")

        return None