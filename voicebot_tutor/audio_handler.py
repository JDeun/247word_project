import pygame
import os
from openai import OpenAI
import config

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=config.OPENAI_API_KEY)

class AudioHandler:
    def __init__(self):
        # Pygame 믹서 초기화
        pygame.mixer.init()

    def play_audio(self, file_path):
        """
        주어진 파일 경로의 오디오를 재생합니다.
        """
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # 오디오 재생이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()

    def text_to_speech(self, text):
        """
        주어진 텍스트를 음성으로 변환하고 재생합니다.
        """
        response_audio = client.audio.speech.create(
            model=config.TTS_MODEL,
            voice=config.VOICE_OPTION,
            input=text,
            speed=config.TTS_SPEED
        )
        # 음성을 파일로 저장
        response_audio.stream_to_file("output.mp3")
        # 저장된 파일 재생
        self.play_audio("output.mp3")
        # 재생 후 임시 파일 삭제
        os.remove("output.mp3")