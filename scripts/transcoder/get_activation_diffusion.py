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
        prompt_embeds, _, _ = pipe.encode_prompt(
            prompts, device=device, num_images_per_prompt=1, max_sequence_length=512
        )
        toks2 = pipe.tokenizer_2(
            prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        attn = toks2["attention_mask"].to(device=device).bool()

    elif model == "sd3":
        prompt_embeds, _, _, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            device=device,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
        )

        toks_clip_1 = pipe.tokenizer(
            prompts, padding="max_length",
            max_length=pipe.tokenizer_max_length, truncation=True, return_tensors="pt"
        )
        toks_clip_2 = pipe.tokenizer_2(
            prompts, padding="max_length",
            max_length=pipe.tokenizer_max_length, truncation=True, return_tensors="pt"
        )
        mask_clip = (toks_clip_1["attention_mask"] | toks_clip_2["attention_mask"]).to(device=device)

        toks_t5 = pipe.tokenizer_3(
            prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        mask_t5 = toks_t5["attention_mask"].to(device=device)

        attn = torch.cat([mask_clip, mask_t5], dim=1).bool()

    else:
        raise ValueError(f"Unknown model '{model}'")

    prompt_embeds = prompt_embeds.to(device=device).contiguous()

    B, T, _ = prompt_embeds.shape
    if attn.shape != (B, T):
        raise RuntimeError(f"Mask/embeds shape mismatch: attn {tuple(attn.shape)} vs embeds {(B, T)}")

    return prompt_embeds, attn


def collect_activations_transformer(
    pipe,
    model,
    prompts,
    batch_size,
    device,
    dtype
):
    logger = logging.getLogger(__name__)

    context_embedder = pipe.transformer.context_embedder

    activations = {
        "context_embedder_in": [],
        "context_embedder_out": [],
    }

    idxs = list(range(len(prompts)))
    random.seed(42)
    random.shuffle(idxs)
    total = len(prompts)

    logger.info("Encoding %d prompts (batch_size=%d)...", total, batch_size)

    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        batch_prompts = [prompts[idxs[k]] for k in range(i, j)]

        prompt_embeds, attn = get_prompt_embeds(
            pipe, model, batch_prompts, device, dtype
        )

        B, T, _ = prompt_embeds.shape
        assert attn.shape == (B, T) and attn.dtype == torch.bool

        valid_in_list, valid_out_list = [], []
        for b in range(B):
            m = attn[b]
            if m.any():
                valid_in_list.append(prompt_embeds[b][m])
                valid_out_list.append(ctx_out[b][m])

        if valid_in_list:
            ctx_in_pack  = torch.vstack(valid_in_list).detach().cpu()
            ctx_out_pack = torch.vstack(valid_out_list).detach().cpu()
        else:
            ctx_in_pack  = torch.empty((0, prompt_embeds.size(-1)), dtype=dtype)
            ctx_out_pack = torch.empty((0, ctx_out.size(-1)), dtype=dtype)

        activations["context_embedder_in"].append(ctx_in_pack)
        activations["context_embedder_out"].append(ctx_out_pack)

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
