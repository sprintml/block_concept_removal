import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
from tqdm.auto import tqdm
import logging
from dataclasses import asdict
import numpy as np
import math

from scripts.transcoder.config import TrainingConfig

from scripts.transcoder.model_training import Transcoder

import wandb

class TranscoderTrainer:
    """Transcoder trainer
    """
    def __init__(self, cfg: TrainingConfig, dataset: Dataset):
        """
        :params cfg: configuration settings for the training
        :params dataset: self explanatory
        """
        self.cfg = cfg
        self.dataset = dataset

        self.num_examples = len(dataset)

        self.device = torch.device(cfg.device)
        self.batch_size = self.cfg.batch_size
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Configuration file: {asdict(cfg)}")
        
        # transcoder dict
        example_x, example_y = dataset[0]
        n_inputs = example_x.shape[-1]
        n_outputs = example_y.shape[-1]

        self.transcoder = Transcoder(
            n_inputs,
            n_outputs,
            cfg.expansion,
            cfg.k,
            cfg.dead_feature_threshold,
            self.device
        )

        # optimizer
        self.optimizer = Adam(
            [{'params': self.transcoder.parameters(), 'lr': 2e-4 / (self.transcoder.n_latents / (2**14)) ** 0.5}],
            eps=cfg.eps,
            betas=(cfg.beta_1, cfg.beta_2),
            fused=cfg.fused,
        )

        self.make_run_name()

        # wandb setup
        if cfg.log_to_wandb:

            wandb.login(key=os.environ["WANDB_TOKEN"])

            wandb.init(
                name=self.run_name,
                project=cfg.wandb_project,
                config=asdict(cfg),
                save_code=False,
            )

    def fit(self):
        """Training script
        """
        # enable full precision
        torch.set_float32_matmul_precision("high")

        # dataloaders
        n_train = int(self.num_examples * 0.9)
        n_val = self.num_examples - n_train

        train_ds, val_ds = torch.utils.data.random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=min(self.batch_size, max(1, len(val_ds))),
            shuffle=False,
            drop_last=True,
        )

        # keep track
        num_batches = (n_train // self.batch_size) * self.cfg.num_epochs
        steps = 0

        # scheduler
        warmup_ratio =  0.03
        warmup_steps = int(warmup_ratio * num_batches)

        def warmup_cosine_lambda(step: int):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            progress = (step - warmup_steps) / float(num_batches - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine

        scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_cosine_lambda)

        device = self.device
        transcoder = self.transcoder

        pbar = tqdm(total=num_batches, desc="Training", initial=steps)
        info = {}

        for _ in range(self.cfg.num_epochs):
            # Training loss
            transcoder.train()

            for batch in train_loader:

                self.optimizer.zero_grad(set_to_none=True)

                total_loss = 0.0
                
                x, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                out = transcoder(x, y)
                
                loss = (
                    out.fvu
                    + self.cfg.auxk_alpha * out.auxk_loss
                    + out.multi_topk_fvu / 8
                )

                if self.cfg.log_to_wandb and steps % self.cfg.wandb_log_frequency == 0:
                    current_lr = scheduler.get_last_lr()[0]

                    dead_mask = transcoder.dead_mask()
                    percentage_dead = dead_mask.sum().item() / transcoder.n_latents

                    info.update(
                        {
                            f"percentage_dead": percentage_dead,
                            f"fvu": out.fvu.float(),
                            f"auxk_loss": out.auxk_loss.float(),
                            f"multi_topk_fvu": out.multi_topk_fvu.float(),
                            f"lr": current_lr
                        }
                    )

                total_loss += loss

                total_loss.backward()

                for pg in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(pg['params'], self.cfg.clip_grad)

                self.optimizer.step()
                scheduler.step()

                if self.cfg.log_to_wandb and steps % self.cfg.wandb_log_frequency == 0:
                    wandb.log(info, step=steps)

                steps += 1
                pbar.update(1)

            # Validation loss
            transcoder.eval()
            transcoder.reset_activation_frequency()

            with torch.no_grad():
                val_fvu = 0.0
                info_eval = {}

                for val_batch in val_loader:
                    x, y = val_batch
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                    out = transcoder(x, y)

                    val_fvu += out.fvu.float() / len(val_loader)

                if self.cfg.log_to_wandb and steps % self.cfg.wandb_log_frequency == 0:
                    counts = transcoder.activation_frequency.cpu().numpy().astype(np.float32)
                    total = counts.sum()
                    freqs = counts / (total + 1e-12)

                    info_eval[f"activation_freq"] = wandb.Histogram(freqs)

                    dead_thresh = 1
                    num_dead = int((counts < dead_thresh).sum())
                    info_eval[f"dead_latents"] = num_dead / transcoder.n_latents

                    info_eval[f"val_fvu"] = val_fvu
                
                wandb.log(info_eval, step=steps)

        pbar.close()

        if self.cfg.log_to_wandb:
            wandb.finish()

    def make_run_name(self):
        """Make the name of the run
        """
        base_dir = os.environ["BASE_DIR"]
        os.makedirs(base_dir, exist_ok=True)

        # get the number of directories
        existing = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
        ]

        # parse the numbers
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_", 1)[1]))
            except ValueError:
                pass
        next_run = max(nums, default=0) + 1

        # make the new run directory
        self.run_name = f"run_{next_run:02d}"

    def save(self, typ=None, name=None):
        """Save the Transcoder to disk
        """
        base_dir = os.environ["BASE_DIR"]

        # make the new run directory
        if name is None:
            self.make_run_name()
        else:
            self.run_name = name
        run_dir = os.path.join(base_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=True)

        self.logger.info(f"Saving checkpoint to {self.run_name!r}")

        fn = os.path.join(run_dir, f"{typ}.pt")

        torch.save(self.transcoder.state_dict(), fn)
        self.logger.info(f"Saved -> {fn}")

        # save training configuration
        cfg_path = os.path.join(run_dir, "configuration.json")
        self.cfg.save_json(cfg_path)
        self.logger.info(f"saved config -> {cfg_path}")

        self.logger.info("Checkpoint saved successfully.")

        return self.run_name

