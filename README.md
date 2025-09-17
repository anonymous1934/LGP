# Local & Global Patching (LGP): Position‑Free Visual Backbones

> 🚀 **Unbelievable yet simple:** On **ImageNet‑100**, adding LGP to the **most basic** backbones yields **+8.40 Top‑1** on **ViT‑T** and **+5.31 Top‑1** on **ViM‑T** with **near‑zero extra compute** and only a **tiny plug‑in** at patch embedding.  
> 👉 We release **pretrained weights** and **training logs** for both models:  
> • **ViT‑T + LGP (IN‑100)** — [Weights](<link-to-vit-t-weights>) · [Logs](<link-to-vit-t-logs>)  
> • **ViM‑T + LGP (IN‑100)** — [Weights](<link-to-vim-t-weights>) · [Logs](<link-to-vim-t-logs>)

**TL;DR.** LGP removes **all position embeddings** in ViT/ViM by enriching each patch token with **multi‑band global wavelet features**. Every token carries an implicit “position signature,” making the encoder **robust to patch order** and often **more accurate**—with **negligible compute overhead**. *(Paper link: <link-to-paper>)*

<p align="center">
  <img src="assets/teaser_lgp.png" alt="LGP vs. traditional patching (teaser)" width="85%">
</p>

*Teaser.* Conventional patching discards spatial structure and relies on learned positional channels later; **Local‑Global Patching** preserves structure by fusing **global multi‑band features** into each patch **at tokenization time**. No separate position learning is required.

---

## Table of Contents
- [Highlights](#highlights)
- [Method at a Glance](#method-at-a-glance)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Drop‑in Integration (PyTorch Pseudocode)](#drop-in-integration-pytorch-pseudocode)
- [Cross‑Framework Usage](#cross-framework-usage)
- [Project Layout](#project-layout)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Position‑free tokens.** LGP equips every patch with a compact summary of the **whole image** across multiple frequency bands, making the encoder **order‑robust** (highly tolerant to patch permutations).  
- **Plug‑and‑play.** Add a lightweight **global wavelet branch** + **Adaptive Fusion Block (AFB)** at the **patch embedding** stage; the rest of ViT/ViM remains unchanged.  
- **Consistent gains.** On ImageNet‑100, the smallest models benefit most—e.g., **ViT‑T +8.40** and **ViM‑T +5.31** top‑1. On ImageNet‑1K, ViT‑S **+1.05** and ViM‑S **+1.39** without retuning.  
- **Negligible overhead.** For typical configs (e.g., `d ≤ 768`, patch size 16), LGP adds **<5% FLOPs** in practice and scales **linearly** with image area.

---

## Method at a Glance

**Two synchronized paths:**
1) **Local** — standard ViT/ViM patch embedding to create local tokens.  
2) **Global** — an `n`‑level **2‑D wavelet‑packet transform** (WPT) of the full image yields **4ⁿ** sub‑bands; sub‑bands are channel‑concatenated, projected with `1×1` conv, and refined by **depthwise‑separable** convs to align with the patch grid.  
3) **Adaptive Fusion Block (AFB)** — a learnable **channel gate** blends local and global signals; three depthwise‑separable convs refine the mixture. The fused tokens already contain multi‑scale semantics and **implicit positional cues**, so encoders need **no absolute/relative position embeddings**.

**Injective positional signature (intuition).** Because wavelet bases are spatially localized, the spectral vector aggregated for patch `(i, j)` is a distinctive signature of its location, yielding strong **order robustness** without explicit positional channels.

**Recommended depth.** Choose `n` so that **`4^n = P_h × P_w`**, where `P_h × P_w` is the patch grid. For square grids (`P_h = P_w = P`), this reduces to **`n = log₂ P`**—enabling patch‑wise spectral pooling.

---

## Results

**ImageNet‑100** (*scaling study*)

| Backbone | Params (M) | GFLOPs | Top‑1 ↑ |
|---|---:|---:|---:|
| **ViM‑T** | 6.42 → 6.65 | 0.16 → 0.21 | **75.10 → 80.41** (+5.31) |
| **ViM‑S** | 23.91 → 24.58 | 0.33 → 0.47 | **79.12 → 84.50** (+5.38) |
| **ViM‑B** | 92.05 → 94.28 | 0.65 → 1.10 | **83.60 → 86.33** (+2.73) |
| **ViT‑T** | 5.72 → 6.06 | 0.91 → 0.98 | **67.64 → 76.04** (+8.40) |
| **ViT‑S** | 22.05 → 23.18 | 3.22 → 3.46 | **74.60 → 80.34** (+5.74) |
| **ViT‑B** | 86.57 → 90.59 | 12.02 → 12.84 | **76.23 → 80.62** (+4.39) |

**Robustness to shuffling.** LGP sharply reduces sensitivity to **patch order** and **band order** (e.g., ViM‑S drop: **−12.66% → −0.12%**; ViT‑S: **−4.29% → −0.09%**).

**ImageNet‑1K.** Out‑of‑the‑box, LGP yields **~+1%** absolute top‑1 at this scale (no extra tuning).

> 🔗 **Artifacts.** We provide **pretrained weights** and **training logs** for **ViT‑T + LGP** and **ViM‑T + LGP** on ImageNet‑100:  
> • ViT‑T + LGP — [Weights](<link-to-vit-t-weights>) · [Logs](<link-to-vit-t-logs>)  
> • ViM‑T + LGP — [Weights](<link-to-vim-t-weights>) · [Logs](<link-to-vim-t-logs>)
