import json
import logging
import re
from openai import OpenAI
from config import Config

class EnglishMaterialGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_name = Config.MODEL_NAME
        with open(Config.PROMPT_FILE, 'r', encoding='utf-8') as f:
            self.prompt_content = f.read()
        logging.info("EnglishMaterialGenerator initialized")

    def generate_material(self, sentences, words):
        try:
            logging.info("Starting to generate learning material")
            
            # 프롬프트 준비
            system_message, user_message = self.prompt_content.split("[사용자 메시지]")
            system_message = system_message.replace("[시스템 메시지]\n", "").strip()
            user_message = user_message.strip()

            formatted_user_message = user_message.format(
                sentences="\n".join(f"- {sentence}" for sentence, _ in sentences),
                words=", ".join(word for word, _ in words)
            )

            logging.info("Prepared prompt for GPT model")
            logging.debug(f"Formatted user message: {formatted_user_message[:500]}...")  # Log first 500 chars

            # GPT API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": formatted_user_message}
                ],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                top_p=Config.TOP_P,
                frequency_penalty=Config.FREQUENCY_PENALTY,
                presence_penalty=Config.PRESENCE_PENALTY
            )

            logging.info("Received response from GPT model")

            # API 응답에서 콘텐츠 추출 및 정제
            content = response.choices[0].message.content
            logging.info(f"Raw content: {content[:1000]}...")  # Log the first 1000 characters of the raw content

            # JSON 추출 및 정제
            json_content = self.extract_json(content)

            logging.info(f"Extracted JSON content: {json_content[:1000]}...")  # Log the first 1000 characters of the extracted JSON

            # JSON 파싱
            try:
                material = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Problematic JSON content: {json_content}")
                # 부분적으로 파싱 시도
                material = self.partial_json_parse(json_content)

            if not material or not isinstance(material, dict) or 'dialogue' not in material or 'vocabulary' not in material:
                raise ValueError("Invalid material structure")

            logging.info("Successfully parsed GPT response into JSON")

            return material

        except Exception as e:
            logging.error(f"Error in generate_material: {str(e)}")
            # 최소한의 유효한 구조 반환
            return {
                "dialogue": [],
                "vocabulary": [],
                "error": str(e),
                "partial_content": content
            }

    def extract_json(self, content):
        """JSON 콘텐츠를 추출하고 정제하는 메서드"""
        # JSON 시작과 끝 찾기
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_content = content[json_start:json_end]
            # 불완전한 JSON 처리
            if json_content.count('{') > json_content.count('}'):
                json_content += '}'  # 닫는 중괄호 추가
            elif json_content.count('{') < json_content.count('}'):
                json_content = '{' + json_content  # 여는 중괄호 추가
            return json_content
        else:
            raise ValueError("No valid JSON found in the response")

    def partial_json_parse(self, json_str):
        """부분적으로 JSON을 파싱하는 메서드"""
        try:
            # dialogue와 vocabulary 부분을 분리
            dialogue_match = re.search(r'"dialogue":\s*\[(.*?)\]', json_str, re.DOTALL)
            vocabulary_match = re.search(r'"vocabulary":\s*\[(.*?)\]', json_str, re.DOTALL)
            
            dialogue = []
            vocabulary = []
            
            if dialogue_match:
                dialogue_str = dialogue_match.group(1)
                dialogue_items = re.findall(r'\{(.*?)\}', dialogue_str, re.DOTALL)
                for item in dialogue_items:
                    try:
                        dialogue.append(json.loads('{' + item + '}'))
                    except:
                        pass
            
            if vocabulary_match:
                vocabulary_str = vocabulary_match.group(1)
                vocabulary_items = re.findall(r'\{(.*?)\}', vocabulary_str, re.DOTALL)
                for item in vocabulary_items:
                    try:
                        vocabulary.append(json.loads('{' + item + '}'))
                    except:
                        pass
            
            return {"dialogue": dialogue, "vocabulary": vocabulary}
        except Exception as e:
            logging.error(f"Error in partial JSON parsing: {str(e)}")
            return {"dialogue": [], "vocabulary": []}