import logging
from openai import OpenAI
from config import Config

class AudioProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def transcribe_audio(self, audio_file):
        try:
            response = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return response.text
        except Exception as e:
            logging.error(f"Error occurred while transcribing audio: {e}")
            return None