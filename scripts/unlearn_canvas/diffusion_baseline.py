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

from huggingface_hub import login
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline
from scripts.utils import setup_logger, get_module_by_name
from scripts.transcoder.model_inference import Transcoder
from scripts.transcoder.model_inference_ca import TranscoderCA
from constants.const import theme_available, class_available

MODEL_PATHS = {
    "sd3":   os.environ["SD3_PATH"],
    "flux":  os.environ["FLUX_PATH"],
}

def load_pipeline(model: str, device: torch.device):
    path = MODEL_PATHS[model]
    if model == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(path, torch_dtype=torch.float16)
    elif model in {"sd1", "sd2", "sd1_u"}:
        pipe = StableDiffusionPipeline.from_pretrained(
            path,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None
        )
    elif model == "flux":
        pipe = FluxPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(model)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", choices=MODEL_PATHS.keys(), required=True)
    parser.add_argument("--seeds",          nargs='+', type=int, default=[188, 288, 588, 688, 888], help="Random seeds for sampling")
    parser.add_argument("--output",         type=str, default="output", help="Root output directory")
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.output, exist_ok=True)
    setup_logger(log_dir=os.path.join(args.output, 'logs'))
    logger = logging.getLogger(__name__)
    logger.info("Arguments: %s", args)

    login(os.environ["HF_TOKEN"])

    model_dir = MODEL_PATHS[args.model]
    logging.info(f"Loading generation model: {args.model} from {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.model, device)

    # Generate and save images
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                with torch.no_grad():
                    img = pipe(prompt).images[0]
                filename = f"{test_theme}_{object_class}_seed{seed}.jpg"
                img.save(os.path.join(out_dir, filename))

    logger.info(f"All images saved to {out_dir}")

if __name__ == "__main__":
    main()
