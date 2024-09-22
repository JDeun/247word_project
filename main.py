from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from audio_processor import AudioProcessor
from text_processor import TextProcessor
from english_material_generator import EnglishMaterialGenerator
from config import Config

app = FastAPI()

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML 템플릿
templates = Jinja2Templates(directory="templates")

audio_processor = AudioProcessor()
text_processor = TextProcessor()
english_generator = EnglishMaterialGenerator()

class LearningMaterial(BaseModel):
    dialogue: list
    vocabulary: list

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/", response_model=str)
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()
    text = audio_processor.transcribe_audio(contents)
    return text

@app.post("/generate_material/", response_model=LearningMaterial)
async def generate_material(file: UploadFile = File(...)):
    contents = await file.read()
    text = audio_processor.transcribe_audio(contents)
    if text:
        sentences, words = text_processor.filter_text(text)
        top_sentences = text_processor.get_top_items(sentences, Config.NUM_SENTENCES)
        top_words = text_processor.get_top_items(words, Config.NUM_WORDS)
        material = english_generator.generate_material(top_sentences, top_words)
        return material
    return {"dialogue": [], "vocabulary": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)