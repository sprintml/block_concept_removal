#!/usr/bin/env python3
import os
import logging
import random
import torch
from scripts.utils import load_prompts, setup_logger, get_module_by_name
from scripts.infinity.inference import InfinityInference, encode_prompt
import argparse
import time

from constants.const import class_available, theme_available

@torch.no_grad()
def collect_activations_direct(
    inf: 'InfinityInference',
    prompts: list[str],
    max_prompts: int,
    batch_size: int,
):
    logger = logging.getLogger(__name__)

    inf.to(device='cuda', dtype=torch.bfloat16)

    activations = {
        "text_proj_for_ca_in":  [],
        "text_proj_for_ca_out": [],
    }

    idxs = list(range(len(prompts)))
    random.seed(42)
    random.shuffle(idxs)
    total = min(len(prompts), max_prompts)
    logger.info("Encoding %d prompts (batch_size=%d)...", total, batch_size)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_idxs    = idxs[start:start + batch_size]
            batch_prompts = [prompts[i] for i in batch_idxs]

            hidden, lengths, cu_seqlens, max_seqlen = encode_prompt(
                inf.text_tokenizer,
                inf.text_encoder,
                batch_prompts
            )

            x = hidden.to(device='cuda', dtype=torch.bfloat16)
            x_in  = inf.infinity.text_norm(x)
            x_out = inf.infinity.text_proj_for_ca(x_in)

            activations["text_proj_for_ca_in"].append(x_in.detach().cpu().contiguous())
            activations["text_proj_for_ca_out"].append(x_out.detach().cpu().contiguous())

            if start % (10 * batch_size) == 0:
                logger.info(f"Processed up to prompt #{start}")

    for k in activations:
        activations[k] = torch.cat(activations[k], dim=0)

    return activations


if __name__ == "__main__":
    t_prog0 = time.perf_counter()

    parser = argparse.ArgumentParser(description="Collect text_proj_for_ca input/output activations")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompt file")
    parser.add_argument("--output_path", type=str, required=True, help="")
    parser.add_argument("--max-prompts", type=int, default=100000, help="Max number of prompts to run")
    parser.add_argument("--batch_size", type=int, default=3, help="Inference batch size")
    parser.add_argument("--out", type=str, default="activations.pt", help="Filename for each .pt dump")
    parser.add_argument("--log-dir", type=str, default="log/collect_activations/", help="Directory for log files")
    parser.add_argument("--model_path", type=str, default="", help="Infinity model checkpoint path")
    parser.add_argument("--vae_path", type=str, default="", help="VAE checkpoint path")
    parser.add_argument("--text_path", type=str, default="", help="Text encoder checkpoint path")
    parser.add_argument("--model_type", type=str, default="", help="Infinity model type")
    parser.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "1M"], help="Precision mode")

    args = parser.parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("=== Program start ===")
    logger.info("Arguments:\n" + "\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    t_load0 = time.perf_counter()
    # Load model
    model = InfinityInference(
        model_path=args.model_path,
        vae_path=args.vae_path,
        text_path=args.text_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        pn=args.pn,
    ).eval()

    t_load1 = time.perf_counter()
    logger.info("Loaded model(s) in %.3fs", t_load1 - t_load0)

    t_prompts0 = time.perf_counter()
    all_prompts = []

    theme_available = ["Cartoon",
        "Cubism",
        "Winter",
        "Pop Art",
        "Ukiyoe",
        "Impressionism",
        "Byzantine",
        "Van Gogh",
        "Bricks",
        "Watercolor",
    ]

    for class_avail in class_available:
        with open(os.path.join(
                    "prompts/unlearn_canvas",
                    f"sd_prompt_{class_avail}.txt",
                    ),
                    "r",
        ) as prompt_file:

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

    acts = collect_activations_direct(
        model,
        all_prompts,
        args.max_prompts,
        args.batch_size
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
        logger.info(f"Saved {label} → {out_dir}/{args.out}")

    t_save1 = time.perf_counter()
    logger.info("Saved tensors in %.3fs", t_save1 - t_save0)

    logger.info("All done.")
