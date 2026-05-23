import os
import sys
import argparse
import logging
import random
from tqdm import tqdm

import torch
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
from itertools import product

class ActivationDataset(Dataset):
    def __init__(self, activations_path):
        self.activations = torch.load(activations_path, map_location='cpu')

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]

from scripts.utils import setup_logger, get_module_by_name
from scripts.transcoder.model_inference import Transcoder
from scripts.infinity.inference import InfinityInference, encode_prompt
from Infinity.tools.run_infinity import load_tokenizer, load_visual_tokenizer
from constants.const import theme_available, class_available

def replace_ffn_with_transcoder(infinity, state_dict_dir):
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')

    # load pretrained transcoder weights
    sd_text_proj_for_ca = torch.load(os.path.join(state_dict_dir, f"text_proj_for_ca.pt"), map_location='cpu')

    transcoder_text_proj_for_ca = Transcoder.from_state_dict(sd_text_proj_for_ca).to(device=device)

    # wrap and assign
    setattr(infinity.infinity, "text_proj_for_ca", transcoder_text_proj_for_ca)

    return transcoder_text_proj_for_ca


def compute_latent_percentages(infer, transcoder, activation_dir, target_concept, batch_size=4096, num_workers=4):
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')

    def compute_one(prompt, null=False):
        hidden, _, _, _ = encode_prompt(infer.text_tokenizer, infer.text_encoder, [prompt])
        hidden = hidden.to(device=device, dtype=torch.float32)
        hidden = infer.infinity.text_norm(hidden)
        if null:
            hidden = hidden[:1, :]
        else:
            hidden = hidden[:-1, :]
        value, idx = transcoder.get_latents(hidden)

        return value, idx

    value_target, idx_target = compute_one(target_concept)
    value_empty, idx_empty = compute_one('_', null=True)

    return (idx_target, value_target, idx_empty, value_empty)


def main():
    parser = argparse.ArgumentParser(
        description="Infinity inference + conditional shutdown + full prompt sweep"
    )
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--vae_path",       type=str, required=True)
    parser.add_argument("--text_path",      type=str, required=True)
    parser.add_argument("--model_type",     type=str, default="infinity_2b")
    parser.add_argument("--pn",             type=str, default="1M")
    parser.add_argument("--state_dict_dir", type=str, required=True, help="Directory with transcoder weights files")
    parser.add_argument("--target",         type=str, required=True, help="Concept to compute latent target for (e.g. 'cat')")
    parser.add_argument("--seeds",          nargs='+', type=int, default=[188, 288, 588, 688, 888], help="Random seeds for sampling")
    parser.add_argument("--output",         type=str, default="output", help="Root output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
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
        batch_size=args.batch_size,
        pn=args.pn,
    ).eval()

    # Replace FFN modules
    transcoder = replace_ffn_with_transcoder(
        infer,
        state_dict_dir=args.state_dict_dir,
    )

    # Compute latent percentages
    latent_info = compute_latent_percentages(
        infer,
        transcoder,
        args.target,
    )

    device = torch.device('cuda')

    idx_target, value_target, idx_empty, value_empty = latent_info

    latents_targets = torch.tensor(idx_target, dtype=torch.int64, device=device)
    latents_empty = torch.tensor(idx_empty, dtype=torch.int64, device=device)

    value_target = torch.tensor(value_target, dtype=torch.float32, device=device)
    value_empty = torch.tensor(value_empty, dtype=torch.float32, device=device)

    latents_empty_exp = latents_empty.expand_as(latents_targets)  
    src_rows = transcoder.W_dec.data[latents_empty_exp]
    scale    = (value_empty / (value_target + 1e-6)).unsqueeze(-1)  

    transcoder.W_dec.data[latents_targets] = src_rows * scale

    # Generate and save images
    out_dir = args.output

    os.makedirs(out_dir, exist_ok=True)

    pairs = list(product(theme_available, class_available))
    total_images = len(args.seeds) * len(pairs)
    with tqdm(total=total_images, desc="Generating images", unit="image") as gen_bar:
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
                    gen_bar.update(1)

    logger.info(f"All images saved to {out_dir}")


if __name__ == "__main__":
    main()
