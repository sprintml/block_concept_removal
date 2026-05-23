import argparse
import torch
from torch.utils.data import DataLoader

from scripts.transcoder.config import TrainingConfig
from scripts.transcoder.trainer import TranscoderTrainer
import logging
from torch.utils.data import Dataset
from scripts.utils import setup_logger
import os
from dataclasses import asdict

class ActivationPairDataset(Dataset):
    def __init__(self, root_dir, layer, fname="activations_8b_style.pt"):
        x_path = os.path.join(root_dir, f"layer{layer}_in", fname)
        y_path = os.path.join(root_dir, f"layer{layer}_out", fname)
        self.x = torch.load(x_path, weights_only=False)
        self.y = torch.load(y_path, weights_only=False)
        self.length = min(self.x.size(0), self.y.size(0))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx].to(torch.float32), self.y[idx].to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train sparse Transcoders on activation pairs")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing activation subfolders per layer.")
    parser.add_argument("--layers", nargs='+', required=True, help="List of layer names to train (corresponding to subfolder names under data-root).")
    parser.add_argument( "--device", type=str, default="cuda", help="Compute device (e.g. 'cuda' or 'cpu').")
    parser.add_argument( "--k", type=int, required=True, help="Value of L0 sparsity")
    parser.add_argument( "--epoch", type=int, required=True, help="Number of epoch")
    parser.add_argument( "--fname", type=str, required=True, help="fname")
    parser.add_argument( "--dead_threshold", type=int, required=True, help="dead_threshold")
    parser.add_argument("--log-dir",        type=str,   default="log/train_transcoder/",                                                              help="Directory for log files")
    args = parser.parse_args()

    setup_logger(log_dir=args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Parsed arguments: %s", vars(args))
    logger.info(f"file: {os.path.basename(__file__)}")

    # Load training configuration
    cfg = TrainingConfig()
    cfg.k = args.k
    cfg.num_epochs = args.epoch
    cfg.fname = args.fname
    cfg.dead_feature_threshold = args.dead_threshold
    cfg.layers = args.layers

    cfg.wandb_project = 'transcoder'

    logger.info("Parsed config: %s", asdict(cfg))

    # Build a dataset for each specified layer
    dataset_dict = {
        layer: ActivationPairDataset(
            root_dir=args.data_root,
            layer=layer,
            fname=args.fname
        )
        for layer in args.layers
    }

    # Initialize and launch the trainer
    trainer = TranscoderTrainer(cfg, dataset_dict)
    trainer.fit()

    # Save model checkpoints
    trainer.save()

if __name__ == "__main__":
    main()
