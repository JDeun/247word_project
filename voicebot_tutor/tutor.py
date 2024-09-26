from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

class EnglishTutor:
    def __init__(self):
        self.conversation_history = []

    def get_response(self, user_input):
        """
        사용자 입력에 대한 튜터의 응답을 생성합니다.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": config.SYSTEM_MESSAGE},
                *self.conversation_history
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE
        )

        tutor_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": tutor_response})
        
        return tutor_response

    def reset_conversation(self):
        """
        대화 기록을 초기화합니다.
        """
        self.conversation_history = []