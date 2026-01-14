
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import math

# Imports are relative to the 'mdlm' directory
from dataloader import MSADataset, BioTokenizer
from diffusion import MaskedDiffusion
from noise_schedule import get_noise_schedule

# A simple helper module for time embeddings
class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.freqs = nn.Parameter(torch.randn(hidden_size // 2) * 10, requires_grad=False)

    def forward(self, t):
        # t is shape (B,)
        args = t[:, None] * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)

# A Transformer model suitable for sequences
class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=12, num_heads=8, dropout=0.1, seq_len=256*128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.time_embedding = TimestepEmbedding(hidden_size)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, hidden_size))

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.final_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, t):
        # x: (B, L) token ids
        # t: (B,) timesteps
        
        # Embed tokens and time
        token_emb = self.token_embedding(x.long())
        time_emb = self.time_embedding(t)
        
        # Add embeddings
        x_emb = token_emb + self.pos_encoder[:, :x.shape[1], :] + time_emb.unsqueeze(1)
        
        # Pass through transformer
        output = self.transformer_encoder(x_emb)
        
        # Project to vocab
        logits = self.final_proj(output)
        return logits

class ProteinMDLLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.tokenizer = BioTokenizer()

        seq_len = self.cfg.data.max_depth * self.cfg.data.max_length
        backbone = SequenceTransformer(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=self.cfg.model.hidden_size,
            num_layers=self.cfg.model.depth,
            num_heads=self.cfg.model.num_heads,
            seq_len=seq_len
        )

        self.diffusion = MaskedDiffusion(
            model=backbone,
            vocab_size=self.tokenizer.vocab_size,
            mask_id=self.tokenizer.mask_token_id,
            **self.cfg.noise,
        )

    def training_step(self, batch, batch_idx):
        # batch shape: (B, Depth * Length)
        B = batch.shape[0]
        t = torch.randint(0, self.cfg.noise.num_timesteps, (B,), device=self.device).long()
        
        loss = self.diffusion.p_losses(batch, t)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        
        scheduler_cfg = self.cfg.lr_scheduler
        if scheduler_cfg._target_ == "mdlm.lr_scheduler.CosineDecayWarmup":
             # A bit of manual instantiation if hydra config is complex
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.trainer.max_steps - scheduler_cfg.warmup_steps, eta_min=scheduler_cfg.min_lr)
             # This is a simplified version. A full warmup implementation would need a LambdaLR or sequential schedulers.
        else: # Default to a simple scheduler
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.trainer.max_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

@hydra.main(config_path="configs", config_name="protein_config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    # 1. Create Dataset and DataLoader
    dataset = hydra.utils.instantiate(cfg.data)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=True,
    )

    # 2. Create Lightning Module
    model = ProteinMDLLightningModule(cfg)

    # 3. Create Trainer
    callbacks = [hydra.utils.instantiate(c) for c in cfg.trainer.callbacks] if cfg.trainer.callbacks else []
    
    # Manually create logger to set a proper save_dir
    tb_logger = pl.loggers.TensorBoardLogger("logs/protein_msa/", name=cfg.project_name)

    trainer_config = {
        'accelerator': cfg.trainer.accelerator,
        'devices': cfg.trainer.devices,
        'max_steps': cfg.trainer.max_steps,
        'precision': cfg.trainer.precision,
        'gradient_clip_val': cfg.trainer.gradient_clip_val,
        'log_every_n_steps': cfg.trainer.log_every_n_steps,
        'callbacks': callbacks,
        'logger': tb_logger
    }
    trainer = pl.Trainer(**trainer_config)

    # 4. Start Training
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    train()
