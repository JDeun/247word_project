from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import logging
import traceback
from audio_processor import AudioProcessor
from text_processor import TextProcessor
from english_material_generator import EnglishMaterialGenerator
from config import Config

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 설정해야 합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프로세서 및 생성기 초기화
audio_processor = AudioProcessor()
text_processor = TextProcessor()
english_generator = EnglishMaterialGenerator()

# 모델 정의
class DialogueEntry(BaseModel):
    speaker: str
    english: str
    korean: str

class VocabularyEntry(BaseModel):
    word: str
    meaning: str

class LearningMaterial(BaseModel):
    dialogue: Optional[List[DialogueEntry]]
    vocabulary: Optional[List[VocabularyEntry]]
    error: Optional[str] = None
    partial_content: Optional[str] = None

@app.post("/generate_material", response_model=LearningMaterial)
async def create_learning_material(file: UploadFile = File(...)):
    """
    음성 파일을 받아 학습 자료를 생성하고 반환하는 엔드포인트
    
    :param file: 업로드된 음성 파일 (지원 형식: WAV, MP3, M4A, 최대 크기: 25MB)
    :return: 생성된 학습 자료
    """
    temp_file_path = None
    try:
        # 파일 형식 및 크기 검증
        allowed_extensions = ['.wav', '.mp3', '.m4a']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds the limit (25MB)")

        logger.info(f"Received audio file: {file.filename}, size: {file_size} bytes")

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = os.path.abspath(temp_file.name)

        logger.info(f"Temporary file created: {temp_file_path}")

        # 파일이 제대로 저장되었는지 확인
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Temporary file not found: {temp_file_path}")

        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Temporary file size: {file_size} bytes")

        # 음성을 텍스트로 변환
        text = audio_processor.transcribe_audio(temp_file_path)
        if text is None:
            raise ValueError("Failed to transcribe audio")
        
        logger.info("Audio transcription completed")
        logger.info(f"Transcribed text: {text[:100]}...")  # 처음 100자만 로깅

        # 텍스트 전처리 및 학습 자료 생성
        sentences, words = text_processor.filter_text(text)
        logger.info(f"Filtered sentences: {len(sentences)}, words: {len(words)}")
        
        top_sentences = text_processor.get_top_items(sentences, Config.NUM_SENTENCES)
        top_words = text_processor.get_top_items(words, Config.NUM_WORDS)
        
        logger.info("Text processing completed")
        logger.info(f"Top sentences: {len(top_sentences)}, Top words: {len(top_words)}")

        material = english_generator.generate_material(top_sentences, top_words)
        
        if "error" in material:
            logger.error(f"Error in generate_material: {material['error']}")
            return LearningMaterial(
                dialogue=[],
                vocabulary=[],
                error=material['error'],
                partial_content=material.get('partial_content')
            )
        
        logger.info("Learning material generated successfully")
        logger.debug(f"Generated material: {material}")  # 디버그 레벨로 생성된 자료 로깅

        return LearningMaterial(**material)

    except Exception as e:
        logger.error(f"Error in create_learning_material: {str(e)}")
        logger.error(f"Full error: {traceback.format_exc()}")
        return LearningMaterial(
            dialogue=[],
            vocabulary=[],
            error=str(e),
            partial_content=traceback.format_exc()
        )

    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

@app.get("/server_check")
async def server_status_check():
    """서버 상태를 확인하는 엔드포인트"""
    return {"status": "good"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)