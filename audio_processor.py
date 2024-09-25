import whisper
import logging
import torch
import os
import soundfile as sf
import librosa
import traceback
import numpy as np

class AudioProcessor:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        logging.info(f"Whisper model loaded on {self.device}")

    def transcribe_audio(self, audio_file):
        try:
            logging.info(f"Attempting to transcribe file: {audio_file}")
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            logging.info(f"File exists, size: {os.path.getsize(audio_file)} bytes")
            
            # 오디오 파일을 읽어서 numpy 배열로 변환
            audio, sample_rate = sf.read(audio_file)
            
            # 스테레오를 모노로 변환 (필요한 경우)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.T)
            
            # 필요한 경우 샘플 레이트 변환
            if sample_rate != 16000:
                logging.info(f"Converting sample rate from {sample_rate} to 16000")
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # 데이터 타입을 float32로 명시적 변환
            audio = audio.astype(np.float32)
            
            if self.device == "cpu":
                with torch.no_grad():
                    result = self.model.transcribe(audio, fp16=False)
            else:
                result = self.model.transcribe(audio)
            
            logging.info("Transcription completed successfully")
            return result["text"]
        except Exception as e:
            logging.error(f"Error occurred while transcribing audio: {str(e)}")
            logging.error(f"Full error: {traceback.format_exc()}")
            return None