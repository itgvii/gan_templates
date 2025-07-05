import os
import sys
import io

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pathlib import Path

ROOT_DIR = Path()
while '.root_anchor' not in os.listdir(ROOT_DIR):
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))
from src.onnx_generator import gen_skin

# Создаем экземпляр FastAPI
app = FastAPI()

@app.get("/")
def home_page():
    return {"message": "The app works!"}

# Определяем эндпоинт для предсказания зарплаты
@app.post("/gen_skin/")
async def model_predict():
    result = gen_skin()  # PIL.Image
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Определяем эндпоинт для предсказания зарплаты
@app.get("/gen_skin_get/")
async def model_predict_get():
    result = gen_skin()  # PIL.Image
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")