import io
from fastapi.testclient import TestClient
from PIL import Image

import os
import sys
from pathlib import Path

ROOT_DIR = Path()
while '.root_anchor' not in os.listdir(ROOT_DIR):
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))
from app.main import app  # Убедитесь, что ваш файл называется main.py или поправьте импорт

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "The app works!"}

def test_gen_skin_post():
    response = client.post("/gen_skin/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Проверим, что результат действительно изображение PNG
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "PNG"

def test_gen_skin_get():
    response = client.get("/gen_skin_get/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "PNG"

def test_download_skin():
    response = client.get("/download_skin")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    # Проверка заголовка Content-Disposition
    assert "content-disposition" in response.headers
    assert "attachment" in response.headers["content-disposition"]
    assert "filename=\"skin.png\"" in response.headers["content-disposition"]
    
    # Проверка, что это изображение PNG
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "PNG"
