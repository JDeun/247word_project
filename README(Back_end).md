# 영어 학습 자료 생성기 프로젝트

## 프로젝트 개요
이 프로젝트는 음성 파일을 입력받아 영어 학습 자료를 자동으로 생성하는 백엔드 시스템입니다. 사용자가 업로드한 음성 파일을 텍스트로 변환한 후, NLP 기술을 활용해 텍스트를 분석하고, 이를 바탕으로 영어 대화와 어휘 목록을 생성합니다. 이 프로젝트는 Flutter 기반의 모바일 애플리케이션과 연동되어 작동합니다.

## 주요 기능
1. 음성 파일 업로드 및 처리 (WAV, MP3, M4A 형식 지원)
2. Whisper 모델을 이용한 음성-텍스트 변환
3. NLP 기반 텍스트 분석 및 전처리
4. GPT 모델을 활용한 맞춤형 영어 학습 자료 생성
5. RESTful API를 통한 클라이언트 통신
6. Firebase를 활용한 사용자 인증

## 기술 스택
- **백엔드 프레임워크**: FastAPI
- **프로그래밍 언어**: Python 3.12
- **음성 처리**: librosa, soundfile
- **NLP 및 텍스트 처리**: NLTK, regex
- **AI 모델**: OpenAI Whisper, GPT-4o-mini
- **사용자 인증**: Firebase Authentication
- **프론트엔드**: Flutter (협업)
- **API 문서화**: Swagger UI (FastAPI 내장)
- **기타**: uvicorn, pydantic

## 프로젝트 구조
```
project/
│
├── backend/
│   ├── main.py            # FastAPI 애플리케이션 및 라우팅
│   ├── audio_processor.py # 음성 파일 처리 및 변환
│   ├── text_processor.py  # NLP 기반 텍스트 분석 및 전처리
│   ├── english_material_generator.py  # GPT 기반 학습 자료 생성
│   ├── firebase_auth.py   # Firebase 인증 통합
│   ├── config.py          # 환경 설정 및 상수
│   └── requirements.txt
│
├── frontend/              # Flutter 프로젝트 (협업)
│
└── README.md
```

## 핵심 기술 상세

### 1. 음성 처리 (audio_processor.py)
- librosa와 soundfile을 사용한 다양한 오디오 형식 지원
- Whisper 모델을 활용한 고정밀 음성-텍스트 변환

```python
def transcribe_audio(self, audio_file):
    audio, sample_rate = sf.read(audio_file)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    
    result = self.model.transcribe(audio)
    return result["text"]
```

### 2. NLP 기반 텍스트 분석 (text_processor.py)
- NLTK를 활용한 토큰화, 품사 태깅, 개체명 인식
- 정규 표현식을 이용한 텍스트 정제
- 불용어 제거 및 어간 추출을 통한 핵심 내용 추출

```python
def process_text(self, text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    processed_text = [
        self.stemmer.stem(word.lower())
        for (word, tag) in tagged
        if word.lower() not in self.stop_words and tag in self.valid_pos_tags
    ]
    
    return processed_text
```

### 3. Firebase 인증 통합 (firebase_auth.py)
- Firebase Admin SDK를 활용한 토큰 검증
- FastAPI 미들웨어를 통한 인증 처리

```python
from firebase_admin import auth, credentials
import firebase_admin

cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

def verify_token(token: str):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/generate_material")
async def create_learning_material(file: UploadFile = File(...), token: str = Header(...)):
    user_id = verify_token(token)
    # 이후 처리 로직...
```

## 개발 일지

### 1주차: 프로젝트 설정 및 기본 서버 구축
- FastAPI 프로젝트 구조 설계
- 기본 API 엔드포인트 구현 및 Swagger UI 설정
- Firebase 프로젝트 설정 및 서버 연동

### 2주차: 음성 처리 및 NLP 모듈 개발
- Whisper 모델 통합 및 음성-텍스트 변환 로직 구현
- NLTK를 활용한 텍스트 분석 기능 구현
- 정규 표현식을 이용한 텍스트 정제 로직 개발

### 3주차: 학습 자료 생성 모듈 개발 및 프론트엔드 연동
- GPT 모델을 활용한 영어 학습 자료 생성 로직 구현
- Flutter 앱과의 API 연동 테스트
- 오류 처리 및 예외 상황 대응 로직 구현

## 현재 진행 상황 및 향후 계획
현재 백엔드 시스템의 기본적인 기능 구현을 완료했습니다. 주요 개선 사항 및 향후 계획은 다음과 같습니다:

1. 성능 최적화: 대용량 음성 파일 처리 시 응답 시간 개선
2. 보안 강화: HTTPS 적용 및 입력 데이터 검증 강화
3. 오류 처리 고도화: 더 상세한 오류 메시지 및 로깅 시스템 구축
4. 확장성 고려: 향후 기능 추가를 위한 모듈화 개선

## 협업 및 역할
- 백엔드 개발 (본인): FastAPI 서버 개발, AI 모델 통합, Firebase 인증 구현
- 프론트엔드 개발 (협업): Flutter를 이용한 모바일 앱 UI 구현

## 배운 점 및 성장
이 프로젝트를 통해 다음과 같은 백엔드 개발 역량을 향상시켰습니다:
- FastAPI를 활용한 효율적인 API 개발 및 문서화
- AI 모델(Whisper, GPT)과 NLP 기술의 실제 애플리케이션 통합
- Firebase를 이용한 인증 시스템 구현
- 프론트엔드 개발자와의 협업을 통한 전체 시스템 아키텍처 이해

이 프로젝트는 실제 사용 가능한 애플리케이션으로 발전시키기 위해 지속적으로 개선 중입니다. 특히 백엔드 시스템의 안정성과 성능 향상에 주력하고 있습니다.
