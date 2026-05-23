from simple_parsing import Serializable, list_field
from dataclasses import dataclass

@dataclass
class TranscoderConfig(Serializable):
    """
    Transcoder configuration
    """
    n_inputs: int = 512
    expansion: int = 16
    k_value: int = 32

@dataclass
class TrainingConfig(Serializable):
    """
    Training configuration for Transcoder
    """
    k: int = 64
    expansion: int = 16
    lr: float = 1e-4
    batch_size: int = 4096
    eps: float = 6.25e-10
    fused: bool = True
    clip_grad: float = 1.0
    auxk_alpha: float = 1/32
    dead_feature_threshold: int = 10_000_000
    ema_multiplier: float = 0.999
    beta_1: float = 0.9
    beta_2: float = 0.999
    device: str = 'cuda'
    num_epochs: int = 1
    num_workers: int = 8
    prefetch_factor: int = 2
    log_to_wandb: bool = True
    wandb_project: str = 'transcoder'
    wandb_log_frequency: int = 1
    fname: str = None
    layers: list = None
