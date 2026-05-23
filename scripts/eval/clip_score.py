#!/usr/bin/env python3
import os
import sys
import argparse
from collections import defaultdict
from typing import Iterable, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(".")
from constants.const import theme_available, class_available

import clip


def iter_ref_image_records(ref_root: str, exclude: Optional[str]) -> Iterable[Tuple[str, str, str]]:
    if exclude is not None:
        if exclude in theme_available:
            theme_list = [t for t in theme_available if t != exclude]
            class_list = class_available
        else:
            theme_list = theme_available
            class_list = [c for c in class_available if c != exclude]
    else:
        theme_list = theme_available
        class_list = class_available

    seed = [188, 288, 588, 688, 888]

    for theme in theme_list:
        for obj in class_list:
            for individual in seed:
                path = os.path.join(ref_root, f"{theme}_{obj}_seed{individual}.jpg")
                yield path, theme, obj


@torch.no_grad()
def compute_clip_score_reference(
    ref_root: str,
    exclude: Optional[str],
    model_name: str = "ViT-B/32",
    batch_size: int = 128,
    device: Optional[str] = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    groups = defaultdict(list)
    total_images = 0
    for path, theme, obj in iter_ref_image_records(ref_root, exclude):
        if os.path.isfile(path):
            groups[(theme, obj)].append(path)
            total_images += 1

    if total_images == 0:
        raise FileNotFoundError(
            f"No reference images found in {ref_root} after exclusion={exclude!r}"
        )

    sims = []
    for (theme, obj), paths in tqdm(groups.items(), desc="CLIP ref scoring"):
        prompt = f"A {obj} image in {theme.replace('_', ' ')} style."
        text_tokens = clip.tokenize([prompt]).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            image_tensor = torch.stack(images, dim=0).to(device)
            image_feat = model.encode_image(image_tensor)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            sim = (image_feat @ text_feat.t()).squeeze(-1)
            sims.append(sim.float().cpu())

    sims = torch.cat(sims, dim=0)
    return float(sims.mean().item())


def main():
    parser = argparse.ArgumentParser(description="CLIP reference score (exclusion-aware)")
    parser.add_argument(
        "--p1", "--path1",
        dest="ref_root",
        default="./data/unlearn_canvas/",
        help="Path to reference images root (tree: {theme}/{class}/{1..5}.jpg)",
    )
    parser.add_argument(
        "--forget-theme",
        type=str,
        default=None,
        help="Theme OR Class to exclude from the reference set (exact name as in constants).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for CLIP image encoding.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help='Model name for clip.load (e.g., "ViT-B/32", "ViT-L/14").',
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="If provided, save the scalar score with torch.save(float, path).",
    )
    args = parser.parse_args()

    score = compute_clip_score_reference(
        ref_root=args.ref_root,
        exclude=args.forget_theme,
        model_name=args.clip_model,
        batch_size=max(1, args.batch_size),
    )
    print(f"CLIP(ref,text): {score:.6f}")

    if args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(float(score), args.output_path)


if __name__ == "__main__":
    main()
