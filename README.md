# Concept Removal for Frontier Image Generative Models

This repository contains the anonymized code associated with the paper:

**Concept Removal for Frontier Image Generative Models**  
*Accepted at ICML 2026*  

---

## Repository Structure

```
.
├── constants/
│   └── const.py
├── prompts/
│   └── unlearn\_canvas/
├── scripts/
│   ├── eval/
│   ├── infinity/
│   ├── transcoder/
│   └── unlearn\_canvas/
└── src/
```

### `constants/`
- **`const.py`** — Stores names of object and styles.

### `prompts/unlearn_canvas/`
- Contains plain text prompt files used for training.
- Each file (`sd_prompt_<Concept>.txt`) corresponds to one object or scene category (e.g., *Dogs*, *Trees*, *Architectures*).  
- These serve as input prompts when testing concept editing.

### `scripts/`

#### `scripts/eval/`
- **`clip_score.py`** — Computes CLIP-based similarity metrics between generated images and reference text.  
- **`fid.py`** — Computes Frechet Inception Distance (FID) for evaluating image quality.  
- **`llava.py`** — Wrapper for evaluation using LLaVA to assess concept removal.  

#### `scripts/infinity/`
- **`inference.py`** — Inference script for the Infinity autoregressive model. Handles prompt-to-image generation.

#### `scripts/transcoder/`
- **`config.py`** — Configuration utilities for training and evaluation.  
- **`get_activation_diffusion.py` / `get_activation_diffusion_mlp.py`** — Extract transcoders training data from diffusion models.  
- **`get_activation_infinity.py`** — Extract transcoders training data from Infinity models.  
- **`kernels.py`** — Implementation of kernel functions used in training the Transcoder.  
- **`model_inference.py`** — Inference pipeline for applying trained Transcoder modules to generative models.  
- **`model_training.py`** — Core training logic for Transcoder modules.  
- **`train.py`** — Entry point for training.
- **`trainer.py`** — Higher-level orchestration of training loops and evaluation.

#### `scripts/unlearn_canvas/`
- **`diffusion_baseline.py`** — Baseline evaluation script for diffusion models without editing.  
- **`diffusion.py`** — Main unlearning script for diffusion models with Transcoder integration.  
- **`infinity_baseline.py`** — Baseline evaluation for Infinity autoregressive models.  
- **`infinity.py`** — Main unlearning script for Infinity with Transcoder integration.

### `src/`
- Placeholder for external dependencies that must be cloned into this folder.
- In practice, this include:
  - [LLaVA](https://github.com/haotian-liu/LLaVA)  
  - [Infinity](https://github.com/FoundationVision/Infinity)


