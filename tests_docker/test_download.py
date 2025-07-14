import requests
import numpy as np
from PIL import Image
from io import BytesIO


url = "http://localhost:8000/download_skin"  # URL до файла
response = requests.get(url)

# Проверка успешности запроса
if response.status_code == 200:
    # Преобразуем байты в объект Image
    image = Image.open(BytesIO(response.content))

    assert np.array(image).shape == (64, 64, 4)
    
    # Показываем изображение (по желанию)
    # image.show()

    # Сохраняем изображение (по желанию)
    # image.save('downloaded_image.jpg')
else:
    print(f'Ошибка загрузки: {response.status_code}')