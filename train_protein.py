import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from dataloader import MSADataset, BioTokenizer
from diffusion import Diffusion 

@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    
    tokenizer = BioTokenizer()
    
    dataset = hydra.utils.instantiate(cfg.data)
    
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=True,
    )

    print(f"Initializing Diffusion model with backbone: {cfg.backbone}")
    model = Diffusion(cfg, tokenizer=tokenizer)

    callbacks = []
    if cfg.trainer.get('callbacks'):
        for _, cb_conf in cfg.trainer.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))
            
    # Logger instantiation
    logger = pl.loggers.TensorBoardLogger("logs/", name="protein_diffusion")

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger
    )

    # 6. 학습 시작
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    train()