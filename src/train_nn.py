import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from lightning import Trainer, seed_everything

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


@hydra.main(version_base='1.3', config_path='../conf', config_name='train_nn.yaml')
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    
    # torch settings
    seed_everything(cfg.general.random_seed)
    torch.set_float32_matmul_precision(cfg.torch_matmul_precision)

    # DataModule
    data_module = instantiate(cfg.data)
    # compute parameters which must be passed to the model
    data_module.setup(stage='train')
    cfg.model = {**cfg.model, **data_module.__getstate__().get('model_hyperparams', {})}
    data_report = data_module.__getstate__().get('data_report', {})

    # Model
    model = instantiate(cfg.model)

    # loggers
    loggers = [instantiate(logger_conf) for logger_conf in cfg.loggers.values()]

    callbacks = [instantiate(callback) for callback in cfg.trainer.callbacks.values()]
    
    # trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epoch,
        accelerator=cfg.trainer.accelerator,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # log hparams
    for logger in trainer.loggers:
        logger.log_hyperparams({'model': cfg.model, 'data': cfg.data, 'data_report': data_report, 'trainer': cfg.trainer})
    
    #fit
    trainer.fit(model, data_module, ckpt_path=cfg.get('ckpt_path'))


if __name__ == '__main__':
    main()