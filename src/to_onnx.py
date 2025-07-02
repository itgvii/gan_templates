import sys
from pathlib import Path
import torch

import sys
import os
from pathlib import Path
ROOT_DIR = Path().resolve()
while '.root_anchor' not in os.listdir(ROOT_DIR):
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))

from src.models.wgan_1 import WGAN


def main():
    # Путь к чекпоинту и куда сохранить ONNX
    ckpt_path = ROOT_DIR / 'outputs/2025-06-23/23-45-13/epoch=99-step=249200.ckpt'
    onnx_path = ROOT_DIR / 'models_onnx/wgan_1.onnx'

    # Загружаем модель из чекпоинта
    model: WGAN = WGAN.load_from_checkpoint(str(ckpt_path))
    model.eval()  # важнo для корректного экспорта
    model.to('cpu')  # ONNX поддерживает только CPU-операторы

    # Параметры — должны совпадать с тем, что вы указывали при обучении
    normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)  # стандартное нормальное распределение
    z = normal_dist.sample((1, model.hparams.latent_dim))
    z = z.view(1, model.hparams.latent_dim, 1, 1).to(model.device)

    model.to_onnx(
        file_path=str(onnx_path),
        input_sample=(z),
        export_params=True,
        opset_version=18,  # или подходящую вам версию,
        # dynamo=True
    )
    print(f"ONNX модель сохранена в {onnx_path}")


if __name__ == '__main__':
    main()
