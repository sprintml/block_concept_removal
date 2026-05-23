import os
import sys
import argparse
import logging
import random
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    def __init__(self, activations_path):
        self.activations = torch.load(activations_path, map_location='cpu')

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]

from huggingface_hub import login
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from scripts.utils import setup_logger, get_module_by_name
from scripts.transcoder.model_inference import Transcoder
from constants.const import theme_available, class_available

MODEL_PATHS = {
    "sd3":   os.environ["SD3_PATH"],
    "flux":  os.environ["FLUX_PATH"],
}

def load_pipeline(model: str):
    path = MODEL_PATHS[model]
    if model == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(path, device_map="balanced")
    elif model == "flux":
        pipe = FluxPipeline.from_pretrained(path, device_map="balanced")
    else:
        raise ValueError(model)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    return pipe


def replace_ffn_with_transcoder(pipe, state_dict_dir):
    logger = logging.getLogger(__name__)
    device = _main_device(pipe)

    # load pretrained transcoder weights
    sd_context = torch.load(os.path.join(state_dict_dir, f"context_embedder.pt"), map_location='cpu')
    sd_text = torch.load(os.path.join(state_dict_dir, f"text_embedder.pt"), map_location='cpu')

    transcoder_context = Transcoder.from_state_dict(sd_context).to(device=device)
    transcoder_text = Transcoder.from_state_dict(sd_text).to(device=device)

    # wrap and assign
    setattr(pipe.transformer, "context_embedder", transcoder_context)
    setattr(pipe.transformer.time_text_embed, "text_embedder", transcoder_text)

    return transcoder_context, transcoder_text

@torch.no_grad()
def get_prompt_embeds(pipe, model: str, prompts):

    def clean_mask(ids: torch.Tensor, mask: torch.Tensor,
                   bos_id=None, eos_id=None, pad_id=None) -> torch.Tensor:
        keep = mask.bool()
        if bos_id is not None:
            keep &= (ids != bos_id)
        if eos_id is not None:
            keep &= (ids != eos_id)
        if pad_id is not None:
            keep &= (ids != pad_id)
        return keep

    if model == "flux":
        prompt_embeds, pooled_embeds, _ = pipe.encode_prompt(prompts)

        toks_t5 = pipe.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        ids_t5, m5 = toks_t5["input_ids"], toks_t5["attention_mask"]

        toks_clip_1 = pipe.tokenizer(
            prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )

        ids_c1, m1 = toks_clip_1["input_ids"], toks_clip_1["attention_mask"]

        mask_c1_clean = clean_mask(
            ids_c1, m1,
            bos_id=getattr(pipe.tokenizer, "bos_token_id", None),
            eos_id=getattr(pipe.tokenizer, "eos_token_id", None),
            pad_id=getattr(pipe.tokenizer, "pad_token_id", None),
        )

        B_print = ids_c1.size(0)
        for b in range(B_print):
            toks1 = pipe.tokenizer.convert_ids_to_tokens(ids_c1[b][mask_c1_clean[b]].tolist())
            print(f"Prompt {b} CLIP-1: {toks1}")

        mask_t5_clean = clean_mask(
            ids_t5, m5,
            bos_id=None,
            eos_id=getattr(pipe.tokenizer_2, "eos_token_id", None),
            pad_id=getattr(pipe.tokenizer_2, "pad_token_id", None),
        )

        attn = mask_t5_clean.to(device=prompt_embeds.device, dtype=torch.bool)

        B_print = ids_t5.size(0)
        for b in range(B_print):
            toks5 = pipe.tokenizer_2.convert_ids_to_tokens(
                ids_t5[b][mask_t5_clean[b]].tolist()
            )
            print(f"Prompt {b} T5:     {toks5}")

    elif model == "sd3":
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            do_classifier_free_guidance=False,
        )

        toks_clip_1 = pipe.tokenizer(
            prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        toks_clip_2 = pipe.tokenizer_2(
            prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        toks_t5 = pipe.tokenizer_3(
            prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
        )

        ids_c1, m1 = toks_clip_1["input_ids"], toks_clip_1["attention_mask"]
        ids_c2, m2 = toks_clip_2["input_ids"], toks_clip_2["attention_mask"]
        ids_t5, m5 = toks_t5["input_ids"], toks_t5["attention_mask"]

        mask_c1_clean = clean_mask(
            ids_c1, m1,
            bos_id=getattr(pipe.tokenizer, "bos_token_id", None),
            eos_id=getattr(pipe.tokenizer, "eos_token_id", None),
            pad_id=getattr(pipe.tokenizer, "pad_token_id", None),
        )
        mask_c2_clean = clean_mask(
            ids_c2, m2,
            bos_id=getattr(pipe.tokenizer_2, "bos_token_id", None),
            eos_id=getattr(pipe.tokenizer_2, "eos_token_id", None),
            pad_id=getattr(pipe.tokenizer_2, "pad_token_id", None),
        )
        mask_t5_clean = clean_mask(
            ids_t5, m5,
            bos_id=None,
            eos_id=getattr(pipe.tokenizer_3, "eos_token_id", None),
            pad_id=getattr(pipe.tokenizer_3, "pad_token_id", None),
        )

        mask_clip = (mask_c1_clean | mask_c2_clean)
        attn = torch.cat([mask_clip, mask_t5_clean], dim=1)
        attn = attn.to(device=prompt_embeds.device, dtype=torch.bool)

        B_print = ids_c1.size(0)
        for b in range(B_print):
            toks1 = pipe.tokenizer.convert_ids_to_tokens(ids_c1[b][mask_c1_clean[b]].tolist())
            toks2 = pipe.tokenizer_2.convert_ids_to_tokens(ids_c2[b][mask_c2_clean[b]].tolist())
            toks5 = pipe.tokenizer_3.convert_ids_to_tokens(ids_t5[b][mask_t5_clean[b]].tolist())
            print(f"Prompt {b} CLIP-1: {toks1}")
            print(f"Prompt {b} CLIP-2: {toks2}")
            print(f"Prompt {b} T5:     {toks5}")

    else:
        raise ValueError(f"Unknown model '{model}' (expected 'flux' or 'sd3').")

    prompt_embeds = prompt_embeds.contiguous()
    B, T, _ = prompt_embeds.shape

    valid_list = []

    for b in range(B):
        m = attn[b]
        if m.any():
            valid_list.append(prompt_embeds[b][m])

    prompt_embeds  = torch.vstack(valid_list)
    prompt_embeds = prompt_embeds.to(dtype=torch.float32)

    pooled_embeds = pooled_embeds.to(dtype=torch.float32)

    return prompt_embeds, pooled_embeds


def compute_latent_percentages(
    pipe,
    transcoder_context,
    transcoder_text,
    model: str,
    target_concept: str,
    model_type: str,
    batch_size: int = 4096,
    num_workers: int = 4):

    logger = logging.getLogger(__name__)
    device = _main_device(pipe)

    prompt_embeds, pooled_embeds = get_prompt_embeds(pipe, model, [target_concept])

    prompt_empty, pooled_empty = get_prompt_embeds(pipe, model, ["_"])

    prompt_empty = prompt_empty[0]

    def _compute_one(
        transcoder,
        subdir: str,
        seed_input: torch.Tensor,
        empty_input: torch.Tensor):

        with torch.no_grad():
            seed_input = seed_input.to(device=device, dtype=torch.float32)
            value_target, idx_target = transcoder.get_latents(seed_input)
            value_empty, idx_empty = transcoder.get_latents(empty_input)

        return (idx_target, value_target, idx_empty, value_empty)

    context_result = _compute_one(
        transcoder_context,
        subdir="context_embedder_in",
        seed_input=prompt_embeds,
        empty_input=prompt_empty
    )

    text_result = _compute_one(
        transcoder_text,
        subdir="text_embedder_in",
        seed_input=pooled_embeds,
        empty_input=pooled_empty
    )

    return {"context": context_result, "text": text_result}

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def _main_device(pipe):
    return next(pipe.transformer.parameters()).device


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--model", choices=MODEL_PATHS.keys(), required=True)
    parser.add_argument("--state_dict_dir", type=str, required=True,
                        help="Directory with {n}_K.pt and {n}_V.pt files")
    parser.add_argument("--target",         type=str, required=True,
                        help="Concept to compute latent target for (e.g. 'cat')")
    parser.add_argument("--seeds",          nargs='+', type=int,
                        default=[188, 288, 588, 688, 888],
                        help="Random seeds for sampling")
    parser.add_argument("--output",         type=str, default="output",
                        help="Root output directory")
    parser.add_argument("--gen_bs", type=int, default=32, help="Batch size for generation")
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.output, exist_ok=True)
    setup_logger(log_dir=os.path.join(args.output, 'logs'))
    logger = logging.getLogger(__name__)
    logger.info("Arguments: %s", args)

    login(os.environ["HF_TOKEN"])

    model_dir = MODEL_PATHS[args.model]
    logging.info(f"Loading generation model: {args.model} from {model_dir}")

    pipe = load_pipeline(args.model)

    # Replace FFN modules
    transcoder_context, transcoder_text = replace_ffn_with_transcoder(pipe=pipe, state_dict_dir=args.state_dict_dir)

    target_text = args.target.replace(" ", "_")

    # Compute latent percentages
    latent_info = compute_latent_percentages(
        pipe,
        transcoder_context,
        transcoder_text,
        args.model,
        target_text,
        args.model
    )

    device = _main_device(pipe)

    transcoders = {"context": transcoder_context, "text": transcoder_text}

    for key, value in latent_info.items():
        transcoder = transcoders[key]

        idx_target, value_target, idx_empty, value_empty = value

        latents_targets = torch.tensor(idx_target, dtype=torch.int64, device=device)
        latents_empty = torch.tensor(idx_empty, dtype=torch.int64, device=device)

        value_target = torch.tensor(value_target, dtype=torch.float32, device=device)
        value_empty = torch.tensor(value_empty, dtype=torch.float32, device=device)

        latents_empty_exp = latents_empty.expand_as(latents_targets)  
        src_rows = transcoder.W_dec.data[latents_empty_exp]
        denom = torch.clamp(value_target, min=1e-6)
        scale    = (value_empty / denom).unsqueeze(-1)  

        transcoder.W_dec.data[latents_targets] = src_rows * scale

    # Generate and save images
    out_dir = args.output

    os.makedirs(out_dir, exist_ok=True)

    # Prepare all jobs up front
    all_jobs = []
    for seed in args.seeds:
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                filename = f"{test_theme}_{object_class}_seed{seed}.jpg"
                all_jobs.append((seed, prompt, filename))

    total_images = len(all_jobs)
    bs = args.gen_bs

    # Single global tqdm across all seeds/batches
    with tqdm(total=total_images, desc="Generating images", unit="image") as gen_bar:
        # seed-stable per-image generators
        generators = {}
        for seed in args.seeds:
            # pre-allocate enough generators; safe upper bound is number of prompts per seed
            n_jobs_seed = sum(1 for s, _, _ in all_jobs if s == seed)
            generators[seed] = [torch.Generator(device=device).manual_seed(seed + i) for i in range(n_jobs_seed)]

        # Run by seed, chunked
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            seed_jobs = [(p, f) for s, p, f in all_jobs if s == seed]
            start = 0
            for batch in chunked(seed_jobs, bs):
                prompts = [p for p, _ in batch]
                gens = generators[seed][start:start+len(batch)]
                start += len(batch)

                with torch.no_grad():
                    out = pipe(prompts, generator=gens)
                    images = out.images

                for (_prompt, fname), img in zip(batch, images):
                    img.save(os.path.join(out_dir, fname))
                    
                gen_bar.update(len(batch))

    logger.info(f"All images saved to {out_dir}")

if __name__ == "__main__":
    main()
