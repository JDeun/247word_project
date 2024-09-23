from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import firebase_admin
from firebase_admin import credentials, auth
from audio_processor import AudioProcessor
from text_processor import TextProcessor
from english_material_generator import EnglishMaterialGenerator
from config import Config
import tempfile
import os

app = FastAPI()

# Firebase 초기화 (인증 목적으로만 사용)
cred = credentials.Certificate("path/to/firebase-credentials.json")
firebase_admin.initialize_app(cred)

# 보안 설정
security = HTTPBearer()

# 프로세서 및 생성기 초기화
audio_processor = AudioProcessor()  # 로컬 Whisper 모델 사용
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
    dialogue: List[DialogueEntry]
    vocabulary: List[VocabularyEntry]

# 사용자 인증 함수
def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@app.post("/generate_material", response_model=LearningMaterial)
async def create_learning_material(
    file: UploadFile = File(...),
    uid: str = Depends(verify_firebase_token)
):
    """
    음성 파일을 받아 학습 자료를 생성하는 엔드포인트
    
    :param file: 업로드된 음성 파일 (지원 형식: WAV, MP3, M4A, 최대 크기: 25MB)
    :param uid: 인증된 사용자의 UID
    :return: 생성된 학습 자료 (LearningMaterial 모델 형식)
    """
    # 파일 형식 및 크기 검증
    allowed_extensions = ['.wav', '.mp3', '.m4a']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    file.file.seek(0, 2)
    if file.file.tell() > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the limit (25MB)")
    file.file.seek(0)

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    try:
        # 음성을 텍스트로 변환 (로컬 Whisper 사용)
        text = audio_processor.transcribe_audio(temp_file_path)
        if text is None:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
        # 텍스트 전처리 및 학습 자료 생성
        sentences, words = text_processor.filter_text(text)
        top_sentences = text_processor.get_top_items(sentences, Config.NUM_SENTENCES)
        top_words = text_processor.get_top_items(words, Config.NUM_WORDS)
        
        material = english_generator.generate_material(top_sentences, top_words)
        if material is None:
            raise HTTPException(status_code=500, detail="Failed to generate learning material")
        
        return material
    finally:
        os.unlink(temp_file_path)

# 서버 상태 확인을 위한 엔드포인트
@app.get("/server_check")
async def server_status_check():
    return {"status": "good"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)