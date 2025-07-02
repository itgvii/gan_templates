import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import sys
import os
from pathlib import Path
ROOT_DIR = Path().resolve()
while '.root_anchor' not in os.listdir(ROOT_DIR):
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

@hydra.main(version_base='1.3', config_path='../conf', config_name='score_generator.yaml')
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    torch.set_float32_matmul_precision(cfg.torch_matmul_precision)

    # load model from checkpoint
    model = instantiate(cfg.model)
    model.eval()

    normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)  # стандартное нормальное распределение
    z = normal_dist.sample((cfg.num_gen_pics, model.hparams.latent_dim))
    z = z.view(cfg.num_gen_pics, model.hparams.latent_dim, 1, 1).to(model.device)
    gen_imgs = model(z)

    real_skin = np.array(Image.open('/home/yan/Downloads/coolzombie.png').convert('RGBA'))
    Path(cfg.save_dir + '/gen_imgs').mkdir(exist_ok=True)

    for i in range(gen_imgs.shape[0]):
        gen_skin = np.array(to_pil_image(gen_imgs[i]))
        transp = (np.array(real_skin[..., 3] != 0, dtype=np.uint8) * 255).reshape(64, 64, 1)
        # transp = np.ones((64, 64, 1), dtype=np.uint8) * 255
        data = (np.concatenate([gen_skin, transp], axis=2))
        result = Image.fromarray(data, 'RGBA')
        
        result.save(cfg.save_dir + f'/gen_imgs/gen_skin_{i}.png')


if __name__ == '__main__':
    main()