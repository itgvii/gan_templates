import os
import sys

from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime import InferenceSession

ROOT_DIR = Path()
while '.root_anchor' not in os.listdir(ROOT_DIR):
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))


# загружаем модель
filename = ROOT_DIR / "models_onnx/wgan_1.onnx"
sess = InferenceSession(filename)

# загружаем real_skin
real_skin = np.array(Image.open(ROOT_DIR / 'app/coolzombie.png').convert('RGBA'))

def gen_skin():
    z = np.random.normal(loc=0.0, scale=1.0, size=(1, 512))
    z = z.reshape(1, 512, 1, 1)
    z = np.array(z, dtype=np.float32)

    # Получаем предсказание от модели
    gen_skin = sess.run(None, {'onnx::ConvTranspose_0': z})[0][0].transpose((1, 2, 0))
    gen_skin = (gen_skin + 1) / 2 * 255
    gen_skin = np.array(gen_skin, dtype=np.uint8)
    transp = (np.array(real_skin[..., 3] != 0, dtype=np.uint8) * 255).reshape(64, 64, 1)
    data = (np.concatenate([gen_skin, transp], axis=2))
    result = Image.fromarray(data, 'RGBA')

    return result


if __name__ == '__main__':
    
    gen_skin().show()