import sys
import os
import argparse
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Union, Tuple, List

from scripts.utils import load_prompts, setup_logger

HERE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.dirname(os.path.dirname(HERE))
SRC       = os.path.join(PROJECT, "src")
SRC_INF   = os.path.join(SRC, "Infinity")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

if SRC_INF not in sys.path:
    sys.path.insert(0, SRC_INF)

from Infinity.tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
)

from Infinity.infinity.utils.dynamic_resolution import (
    dynamic_resolution_h_w,
    h_div_w_templates,
)

class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.prompts: List[str] = load_prompts(path)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def encode_prompt(
    text_tokenizer,
    text_encoder,
    prompts: List[str]
) -> Tuple[torch.FloatTensor, List[int], torch.IntTensor, int]:

    tokens = text_tokenizer(
        text=prompts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokens.input_ids.cuda(non_blocking=True)
    attn_mask = tokens.attention_mask.cuda(non_blocking=True)

    out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    text_features = out['last_hidden_state'].float()

    lengths = attn_mask.sum(dim=-1).cpu().tolist()

    packed = pack_padded_sequence(
        text_features, lengths, batch_first=True, enforce_sorted=False
    )
    kv_compact = packed.data.contiguous()

    lens_tensor = torch.tensor(lengths, dtype=torch.int32, device=attn_mask.device)
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=attn_mask.device),
        lens_tensor.cumsum(0, dtype=torch.int32)
    ], dim=0)

    max_seqlen = max(lengths)

    return kv_compact, lens_tensor, cu_seqlens, max_seqlen


def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


class InfinityInference(nn.Module):
    """
    Simplified Infinity inference module.
    """
    def __init__(
        self,
        model_path: str,
        vae_path: str,
        text_path: str,
        model_type: str,
        batch_size: int = 3,
        pn: str = '1M',
        cfg: float = 4,
        neg_prompt: str = None
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = set_seed(42)
        self.logger.info(f"Using device={self.device}, seed={self.seed}")

        vae_types = {
            "infinity_2b": 32,
            "infinity_8b": 14,
        }

        patchify = {
            "infinity_2b": 0,
            "infinity_8b": 1,
        }

        # bundle args
        self.args = argparse.Namespace(
            pn=pn,
            model_path=model_path,
            model_type=model_type,
            vae_type=vae_types[model_type],
            vae_path=vae_path,
            cfg_insertion_layer=0,
            sampling_per_bits=1,
            bf16=1,
            cache_dir='/cache',
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            text_channels=2048,
            apply_spatial_patchify=patchify[model_type],
            h_div_w_template=1.0,
            use_flex_attn=0,
            checkpoint_type='torch_shard' if os.path.isdir(model_path) else 'torch',
            enable_model_cache=0,
            batch_size=batch_size,
        )

        self.neg_prompt = neg_prompt

        # load models
        self.text_tokenizer, self.text_encoder, self.vae, self.infinity = self._loading_models(self.args, text_path, self.device)

        # prepare scale schedule
        self.scale_schedule, self.cfg_list, self.tau_list = self._prepare_scale_schedule(self.args.pn, dynamic_resolution_h_w, h_div_w_templates, cfg)

        if self.neg_prompt is not None:
            self.neg_prompt = encode_prompt(
                self.text_tokenizer,
                self.text_encoder,
                [self.neg_prompt],
            )

        self.logger.info("Model initialized and ready.")

    def _prepare_scale_schedule(self, pn, dynamic_resolution_h_w, h_div_w_templates, cfg):
        h_div_w_template_ = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - 1.0))
        ]
        scales = dynamic_resolution_h_w[h_div_w_template_][pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scales]
        cfg_list = [cfg] * len(scale_schedule)
        tau_list = [1.0] * len(scale_schedule)

        return scale_schedule, cfg_list, tau_list


    def _loading_models(self, args, text_path, device):
        # load text encoder
        self.logger.info("Loading text encoder...")
        text_tokenizer, text_encoder = load_tokenizer(t5_path=text_path)
        text_encoder.to(device).eval()

        # load VAE
        self.logger.info("Loading VAE...")
        vae = load_visual_tokenizer(args)
        vae.to(device).eval()

        # load Infinity transformer
        self.logger.info("Loading Infinity transformer...")
        infinity = load_transformer(vae, args)
        infinity.to(device).eval()

        return text_tokenizer, text_encoder, vae, infinity


    def forward(self, prompts:Union[List[str], Tuple]):
        self.logger = logging.getLogger(__name__)
        if (
            isinstance(prompts, tuple)
            and len(prompts) == 4
            and isinstance(prompts[0], torch.Tensor)
        ):
            kv_compact, lens_tensor, cu_seqlens, max_seqlen = prompts
        else:
            self.logger.debug(f"Encoding prompt: {prompts!r}")
            kv_compact, lens_tensor, cu_seqlens, max_seqlen = encode_prompt(
                self.text_tokenizer,
                self.text_encoder,
                prompts,
            )

        batch_size = lens_tensor.size(0)

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=True,
            dtype=torch.bfloat16,
            cache_enabled=True,
        ):
            kv_compact = kv_compact.to(torch.bfloat16)
            label = (kv_compact, lens_tensor, cu_seqlens, max_seqlen)

            _, _, images = self.infinity.autoregressive_infer_cfg(
                vae=self.vae,
                scale_schedule=self.scale_schedule,
                label_B_or_BLT=label,
                negative_label_B_or_BLT=self.neg_prompt,
                B=batch_size,
                g_seed=self.seed,
                cfg_list=self.cfg_list,
                tau_list=self.tau_list,
                top_k=900,
                top_p=0.97,
                returns_vemb=1,
                norm_cfg=False,
                cfg_insertion_layer=[self.args.cfg_insertion_layer],
                vae_type=self.args.vae_type,
                ret_img=True,
                trunk_scale=1000,
                sampling_per_bits=self.args.sampling_per_bits,
                inference_mode=True,
            )

        self.logger.debug("Inference step complete, returning image tensor.")
        return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Infinity inference.")
    parser.add_argument("--model_path",     type=str,   default="",                                               help="Infinity model checkpoint path.")
    parser.add_argument("--vae_path",       type=str,   default="",                                           help="VAE checkpoint path.")
    parser.add_argument("--text_path",      type=str,   default='',        help="Text encoder checkpoint path.")
    parser.add_argument("--model_type",     type=str,   default='infinity_2b',                                                                          help="Infinity model type.")
    parser.add_argument("--batch_size",     type=int,   default=3,                                                                                      help="Batch size.")
    parser.add_argument("--prompts",        type=str,   required=True,                                                                                  help="List of prompts for generation.")
    parser.add_argument("--output",         type=str,   default="output/",                                                                              help="Output image filename.")
    parser.add_argument("--max_prompts",    type=int,   default=10,                                                                                     help="Maximum number of prompts to run.")
    parser.add_argument('--pn',             type=str,   default='1M',   choices=['0.06M', '0.25M', '1M'])
    parser.add_argument("--log-dir",        type=str,   default="log/infinity_inference/",                                                              help="Directory for log files")
    args = parser.parse_args()

    # initialize logging
    setup_logger(log_dir=args.log_dir)
    logger = logging.getLogger(__name__)

    logger.info("Arguments:\n" + "\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    # initialize inference model
    logger.info("Starting Infinity inference script.")
    infer = InfinityInference(
        model_path=args.model_path,
        vae_path=args.vae_path,
        text_path=args.text_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        pn=args.pn
    ).eval()

    prompts_dataset = PromptDataset(args.prompts)
    prompts_loader = DataLoader(prompts_dataset, batch_size=args.batch_size, shuffle=False)

    total = len(prompts_dataset)
    logger.info(f"Loaded {total} prompts, will run up to {total} images in batches of {args.batch_size}.")
    os.makedirs(args.output, exist_ok=True)

    start = 0
    max_prompts = args.max_prompts
    for batch_prompts in prompts_loader:
        if start >= args.max_prompts:
            break
        start += args.batch_size

        logger.info(f"Generating batch prompts [{start}:{start + len(batch_prompts)}]")

        imgs = infer.forward(batch_prompts)

        for j, img in enumerate(imgs):
            idx = start + j
            img_np = img.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            filename = os.path.join(args.output, f"{idx:04d}.png")
            cv2.imwrite(filename, img_np)
            logger.info(f"Saved {filename}")

    logger.info("All done. Exiting.")

