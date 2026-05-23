import os
import re
import sys
import argparse
import csv
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from constants.const import class_available, theme_available

HERE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.dirname(os.path.dirname(HERE))
SRC       = os.path.join(PROJECT, "src")
SRC_LLAVA   = os.path.join(SRC, "LLaVA")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

if SRC_LLAVA not in sys.path:
    sys.path.insert(0, SRC_LLAVA)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

def parse_style_class(fn):
    base = os.path.basename(fn)
    name = os.path.splitext(base)[0]
    prefix = re.split(r"_seed", name)[0]
    for cls in class_available:
        if prefix.endswith(f"_{cls}"):
            style_raw = prefix[:-(len(cls)+1)]
            style = style_raw.replace("_", " ")
            return style, cls
    raise ValueError(f"Cannot parse style/class from {fn}")

def is_valid_image(fp: str) -> bool:
    try:
        parse_style_class(fp)
        with Image.open(fp) as im:
            im.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError) as e:
        print(f"[WARN] Dropping {fp}: {e}")
        return False


ALIASES_BASE = {
    "van gogh": "Van Gogh",
    "vangogh": "Van Gogh",
    "pop art": "Pop Art",
    "cubist": "Cubism",
    "ukiyo e": "Ukiyoe",
    "japanese woodblock": "Ukiyoe",
    "impressionist": "Impressionism",
    "byzantium": "Byzantine",
    "byzantium art": "Byzantine",
    "cartoons": "Cartoon",
    "comic": "Cartoon",
    "winter scene": "Winter",
    "snowy": "Winter",
    "brickwork": "Bricks",
    "brick": "Bricks",
    "water colour": "Watercolor",
    "water colour painting": "Watercolor",
}


def norm_text(s: str) -> str:
    s = s.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_label_maps(active_labels):
    norm_to_label = {norm_text(lbl): lbl for lbl in active_labels}
    aliases = {k: v for k, v in ALIASES_BASE.items() if v in active_labels}
    return norm_to_label, aliases

def match_to_subset(free_text: str, active_labels, norm_to_label, aliases) -> str:
    t = norm_text(free_text)
    if not t:
        return "Unknown"
    if t in norm_to_label:
        return norm_to_label[t]
    if t in aliases:
        return aliases[t]
    t2 = re.sub(r"[^\w\s]", "", t).strip()
    if t2 in norm_to_label:
        return norm_to_label[t2]
    if t2 in aliases:
        return aliases[t2]
    import difflib
    candidates = list(norm_to_label.keys()) + list(aliases.keys())
    best = difflib.get_close_matches(t2, candidates, n=1, cutoff=0.6)
    if best:
        b = best[0]
        if b in norm_to_label:
            return norm_to_label[b]
        return aliases[b]
    return "Unknown"

def make_prompt_with_image_tokens(qs, model):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs

def build_conv_prompt(qs, conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def infer_one(tokenizer, model, image_processor, query_text, image_path, conv_mode, max_new_tokens):
    qs = make_prompt_with_image_tokens(query_text, model)
    prompt = build_conv_prompt(qs, conv_mode)

    pil = Image.open(image_path).convert("RGB")
    images_tensor = process_images([pil], image_processor, model.config).to(
        model.device,
        dtype=torch.float16 if getattr(model, "dtype", torch.float16) == torch.float16 else torch.float32
    )
    image_sizes = [pil.size]

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0,
            top_p=1.0,
            num_beams=1,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def mc_prompt(question, labels):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question, "Choose exactly one option and reply with the letter only."]
    mapping = {}
    for i, lab in enumerate(labels):
        lines.append(f"{letters[i]}) {lab}")
        mapping[letters[i]] = lab
    return "\n".join(lines), mapping

def llava_predict(image_path, labels, question, tokenizer, model, image_processor, conv_mode, max_new_tokens):
    prompt, letter_map = mc_prompt(question, labels)
    raw = infer_one(tokenizer, model, image_processor, prompt, image_path, conv_mode, max_new_tokens)
    m = re.search(r"[A-Z]", raw.strip())
    if m and m.group(0) in letter_map:
        return letter_map[m.group(0)], raw
    norm_to_label, aliases = build_label_maps(labels)
    return match_to_subset(raw, labels, norm_to_label, aliases), raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UA IRA CRA evaluation with LLaVA")
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--theme",      required=True)
    parser.add_argument("--task",       required=True, choices=["style", "class"])

    # LLaVA specific args
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode",  type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)

    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if args.conv_mode:
        conv_mode = args.conv_mode
    elif "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    # counters
    UA_tot = UA_corr = 0
    IRA_tot = IRA_corr = 0
    CRA_tot = CRA_corr = 0

    # collect images
    all_fps = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    candidates = []
    for fp in all_fps:
        try:
            style, _ = parse_style_class(fp)
            if style != "Seed_Images":
                candidates.append(fp)
        except ValueError as e:
            print(f"[WARN] Skipping {fp}: {e}")

    imgs = [fp for fp in candidates if is_valid_image(fp)]
    if not imgs:
        raise SystemExit("No valid images to evaluate after pre filtering.")

    # choose target index
    if args.task == "style":
        label_list_task = theme_available
        label_list_other = class_available
    else:
        label_list_task = class_available
        label_list_other = theme_available

    if args.theme not in label_list_task:
        raise SystemExit(f"Theme '{args.theme}' not in label set for task '{args.task}'.")

    targ_idx = label_list_task.index(args.theme)

    # evaluate per image
    for fp in tqdm(imgs, desc=f"LLaVA eval {args.task}={args.theme}"):
        # ground truth from file name
        true_style, true_class = parse_style_class(fp)
        ts = theme_available.index(true_style)
        tc = class_available.index(true_class)

        # predictions from LLaVA
        pred_style, _raw_s = llava_predict(
            fp, theme_available,
            "What is the artistic style of this image.",
            tokenizer, llava_model, image_processor, conv_mode, args.max_new_tokens
        )
        pred_class, _raw_c = llava_predict(
            fp, class_available,
            "What is the object class of this image.",
            tokenizer, llava_model, image_processor, conv_mode, args.max_new_tokens
        )
        sp = theme_available.index(pred_style) if pred_style in theme_available else -1
        cp = class_available.index(pred_class) if pred_class in class_available else -1

        if args.task == "style":
            m_target = (ts == targ_idx)
            if m_target:
                UA_tot  += 1
                UA_corr += int(sp != targ_idx)
            else:
                IRA_tot  += 1
                IRA_corr += int(sp == ts)
                CRA_tot  += 1
                CRA_corr += int(cp == tc)
        else:
            m_target = (tc == targ_idx)
            if m_target:
                UA_tot  += 1
                UA_corr += int(cp != targ_idx)
            else:
                IRA_tot  += 1
                IRA_corr += int(cp == tc)
                CRA_tot  += 1
                CRA_corr += int(sp == ts)

    unlearn_acc    = UA_corr / UA_tot   if UA_tot else float("nan")
    in_domain_acc  = IRA_corr / IRA_tot if IRA_tot else float("nan")
    out_domain_acc = CRA_corr / CRA_tot if CRA_tot else float("nan")

    target_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(target_dir, exist_ok=True)
    out_csv = os.path.join(target_dir, "retain_and_unlearn.csv")

    df = pd.DataFrame([{
        "target": args.theme,
        "unlearn_acc":    unlearn_acc,
        "in_domain_acc":  in_domain_acc,
        "out_domain_acc": out_domain_acc
    }]).set_index("target")
    df.to_csv(out_csv)
    print(f"Wrote {out_csv}")
