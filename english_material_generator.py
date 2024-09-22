import json
import logging
from openai import OpenAI
from config import Config

class EnglishMaterialGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        with open(Config.PROMPT_FILE, 'r', encoding='utf-8') as f:
            self.prompt_content = f.read()

    def generate_material(self, sentences, words):
        try:
            system_message, user_message = self.prompt_content.split("[사용자 메시지]")
            system_message = system_message.replace("[시스템 메시지]\n", "").strip()
            user_message = user_message.strip()

            formatted_user_message = user_message.format(
                sentences="\n".join(f"- {sentence}" for sentence, _ in sentences),
                words=", ".join(word for word, _ in words)
            )

            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": formatted_user_message}
                ],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                top_p=Config.TOP_P,
                frequency_penalty=Config.FREQUENCY_PENALTY,
                presence_penalty=Config.PRESENCE_PENALTY,
                stop=Config.STOP_SEQUENCES
            )

            content = response.choices[0].message.content
            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```").strip()
            if content.startswith("json"):
                content = content.replace("json", "", 1).strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logging.error(f"GPT response is not a valid JSON. Error: {str(e)}")
            logging.error(f"Raw response: {content}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")

        return None