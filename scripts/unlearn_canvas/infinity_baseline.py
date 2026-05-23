import os
import sys
import argparse
import logging
import random

import torch
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    def __init__(self, activations_path):
        self.activations = torch.load(activations_path, map_location='cpu')

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]

from scripts.utils import setup_logger, get_module_by_name
from scripts.transcoder.model_inference import Transcoder
from scripts.transcoder.model_inference_ca import TranscoderCA
from scripts.infinity.inference import InfinityInference, encode_prompt
from Infinity.tools.run_infinity import load_tokenizer, load_visual_tokenizer
from constants.const import theme_available, class_available

def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--vae_path",       type=str, required=True)
    parser.add_argument("--text_path",      type=str, required=True)
    parser.add_argument("--model_type",     type=str, default="infinity_2b")
    parser.add_argument("--pn",             type=str, default="1M")
    parser.add_argument("--seeds",          nargs='+', type=int,
                        default=[188, 288, 588, 688, 888],
                        help="Random seeds for sampling")
    parser.add_argument("--output",         type=str, default="output",
                        help="Root output directory")
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.output, exist_ok=True)
    setup_logger(log_dir=os.path.join(args.output, 'logs'))
    logger = logging.getLogger(__name__)
    logger.info("Arguments: %s", args)

    # Load Infinity model skeleton
    infer = InfinityInference(
        model_path=args.model_path,
        vae_path=args.vae_path,
        text_path=args.text_path,
        model_type=args.model_type,
        batch_size=1,
        pn=args.pn,
    ).eval()

    # Generate and save images
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        infer.seed = seed
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                with torch.no_grad():
                    imgs = infer.forward([prompt])
                img = imgs[0].cpu().numpy()
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                filename = f"{test_theme}_{object_class}_seed{seed}.jpg"
                cv2.imwrite(os.path.join(out_dir, filename), img)

    logger.info(f"All images saved to {out_dir}")

if __name__ == "__main__":
    main()
