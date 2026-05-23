#!/usr/bin/env python3
import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from diffusers import StableDiffusion3Pipeline, FluxPipeline
import time

from constants.const import class_available, theme_available

MODEL_PATHS = {
    "sd3":   os.environ["SD3_PATH"],
    "flux":  os.environ["FLUX_PATH"],
}

def setup_logger(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(log_dir) / "collect_sd_ca_activations.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logging.getLogger().addHandler(console)


def read_prompts(path: str | Path) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


@torch.no_grad()
def get_prompt_embeds(pipe, model, prompts, device, dtype):
    if model == "flux":
        _, pooled_in, _ = pipe.encode_prompt(
            prompts, device=device, num_images_per_prompt=1, max_sequence_length=512
        )

    elif model == "sd3":
        _, _, pooled_in, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            device=device,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
        )

    else:
        raise ValueError(f"Unknown model '{model}'")

    return pooled_in


def collect_activations_transformer(
    pipe,
    model,
    prompts,
    batch_size,
    device,
    dtype
):
    logger = logging.getLogger(__name__)

    text_embedder    = pipe.transformer.time_text_embed.text_embedder

    activations = {
        "text_embedder_in":     [],
        "text_embedder_out":    [],
    }

    idxs = list(range(len(prompts)))
    random.seed(42)
    random.shuffle(idxs)
    total = len(prompts)

    logger.info("Encoding %d prompts (batch_size=%d)...", total, batch_size)

    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        batch_prompts = [prompts[idxs[k]] for k in range(i, j)]

        pooled_in = get_prompt_embeds(
            pipe, model, batch_prompts, device, dtype
        )

        pooled_in  = pooled_in.to(device=device, dtype=dtype)
        pooled_out = text_embedder(pooled_in)

        activations["text_embedder_in"].append(pooled_in.detach().cpu())
        activations["text_embedder_out"].append(pooled_out.detach().cpu())

        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed up to prompt #{j}")

    for k in list(activations.keys()):
        if len(activations[k]) > 0:
            if len(activations[k]) == 1:
                activations[k] = activations[k][0].to(dtype)
            else:
                activations[k] = torch.cat(activations[k], dim=0).to(dtype)
        else:
            activations[k] = torch.empty(0, dtype=dtype)

    return activations


def main():
    t_prog0 = time.perf_counter()

    # parsing
    parse = argparse.ArgumentParser(description="Collect CA activations (inputs, context_embedder, text_embedder) for SD3, FLUX")
    parse.add_argument("--model", type=str, choices=MODEL_PATHS.keys(), required=True, help="sd3, flux")
    parser.add_argument("--output_path", type=str, required=True, help="")
    parse.add_argument("--batch-size", type=int, default=4)
    parse.add_argument("--out", type=str, default="activations.pt")
    parse.add_argument("--log-dir", type=str, default="log/get_activations_diffusion/", help="Directory for log files")
    args = parse.parse_args()

    # logging
    setup_logger(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:\n" + "\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    # configuration
    device = "cuda"
    dtype = torch.bfloat16

    t_load0 = time.perf_counter()

    # model
    model_id = MODEL_PATHS[args.model]
    logger.info(f"Loading {args.model} from {model_id}")

    if args.model == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
    elif args.model == "flux":
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
    else:
        raise ValueError(args.model)
    t_load1 = time.perf_counter()
    logger.info("Loaded model(s) in %.3fs", t_load1 - t_load0)

    t_prompts0 = time.perf_counter()

    all_prompts = []
    
    with open("prompts/coco_train2014_30k_prompts.txt", "r",) as prompt_file:
        for prompt in prompt_file:
            prompt = prompt.strip()
            prompt = prompt if not prompt.endswith(".") else prompt[:-1]
            for theme in theme_available:
                theme_prompt = (
                    f"{prompt} in {theme.replace('_', ' ')} style."
                )
                all_prompts.append(theme_prompt)
            all_prompts.append(prompt + ".")

    t_prompts1 = time.perf_counter()
    logger.info("Built %d prompts in %.3fs", len(all_prompts), t_prompts1 - t_prompts0)

    t_collect0 = time.perf_counter()
    acts = collect_activations_transformer(
        pipe=pipe,
        model=args.model,
        prompts=all_prompts,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_collect1 = time.perf_counter()
    logger.info("Collected activations in %.3fs", t_collect1 - t_collect0)

    t_save0 = time.perf_counter()
    for label, tensor in acts.items():
        out_dir = os.path.join(output_path, label)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(tensor, os.path.join(out_dir, args.out))
        logging.info(f"Saved {label} → {out_dir}/{args.out}")

    t_save1 = time.perf_counter()
    logger.info("Saved tensors in %.3fs", t_save1 - t_save0)

    logger.info("All done.")

if __name__ == "__main__":
    main()
