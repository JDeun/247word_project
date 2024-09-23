import whisper
import logging

class AudioProcessor:
    def __init__(self, model_size="base"):
        """
        AudioProcessor 초기화
        :param model_size: Whisper 모델 크기. 옵션: "tiny", "base", "small", "medium", "large"
        
        모델 크기별 특징:
        - tiny: 가장 작고 빠른 모델. 정확도는 낮지만 리소스 요구 사항이 가장 적음.
        - base: 적당한 정확도와 속도의 균형. 대부분의 경우에 충분한 성능.
        - small: base보다 더 나은 정확도, 약간 더 많은 리소스 필요.
        - medium: 높은 정확도, 더 많은 리소스 필요. 복잡한 오디오에 적합.
        - large: 가장 정확한 모델. 가장 많은 리소스 필요. 다국어 및 복잡한 오디오에 최적.

        모델 크기가 커질수록:
        - 정확도 향상
        - 처리 시간 증가
        - 필요한 메모리 및 계산 리소스 증가
        - 다양한 언어 및 악센트에 대한 이해도 향상
        """
        self.model = whisper.load_model(model_size)

    def transcribe_audio(self, audio_file):
        """
        오디오 파일을 텍스트로 변환
        :param audio_file: 변환할 오디오 파일 경로
        :return: 변환된 텍스트 또는 오류 발생 시 None
        """
        try:
            result = self.model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            logging.error(f"Error occurred while transcribing audio: {e}")
            return None