# Local & Global Patching (LGP): Position‑Free Visual Backbones

> **TL;DR** — LGP removes **all position embeddings** in ViT/ViM by enriching each patch token with **multi‑band global wavelet features**. Every token carries an implicit “position signature,” making the model **invariant to patch order**, more robust, and often more accurate—*with negligible compute overhead*. See the paper: **Local and Global Patching: Goodbye for Good to Position Embeddings in Vision Transformers and Mambas**. :contentReference[oaicite:0]{index=0}

<p align="center">
  <img src="assets/teaser_lgp.png" alt="LGP vs. traditional patching (recreate Fig.1 as a teaser)" width="85%">
</p>

*Figure 1 on page 1* contrasts **conventional patching** (spatial relationships lost then re‑learned via position embeddings) with **Local‑Global Patching** (spatial structure preserved by fusing global multi‑band features into each patch at tokenization time). No extra position learning is required. :contentReference[oaicite:1]{index=1}

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

- **Position‑free tokens.** LGP equips every patch with a compact summary of the **whole image** across multiple frequency bands, making the encoder **permutation‑invariant to patch order**. See Fig.1 (p.1) and the pipeline in Fig.2 (p.4). :contentReference[oaicite:2]{index=2}  
- **Plug‑and‑play.** Add a lightweight **global wavelet branch** + **Adaptive Fusion Block (AFB)** at the **patch embedding** stage; the rest of ViT/ViM remains unchanged. :contentReference[oaicite:3]{index=3}  
- **Consistent gains.** On ImageNet‑100, **ViM‑S 79.12→84.50** and **ViT‑S 74.60→80.34**; smaller models benefit most. On ImageNet‑1K, ViM‑S **+1.39** and ViT‑S **+1.05** without retuning. :contentReference[oaicite:4]{index=4}  
- **Negligible overhead.** For typical configs (e.g., `d ≤ 768`, patch size 16), LGP adds **<5% FLOPs** relative to a single Transformer block, and cost scales **linearly** with image area (Eqs. 13–14, p.5). :contentReference[oaicite:5]{index=5}

---

## Method at a Glance

**Two synchronized paths (Fig.2, p.4):**  
1) **Local** — standard ViT/ViM patch embedding creates local tokens.  
2) **Global** — an `n`‑level **2‑D wavelet‑packet transform** (WPT) of the full image yields **4ⁿ** sub‑bands capturing coarse layout to fine detail; sub‑bands are channel‑concatenated, projected with `1×1` conv, and refined by **depthwise‑separable** convs to align with the patch grid.  
3) **Adaptive Fusion Block (AFB)** — a learnable **channel gate** blends local vs. global; three depthwise‑separable convs refine the mixture. The fused tokens already contain multi‑scale semantics and **implicit position cues**, so encoders need **no absolute or relative position embeddings**. :contentReference[oaicite:6]{index=6}

**Injective positional signature.** Because wavelet bases are spatially localized, the spectral vector aggregated for patch `(i,j)` is **one‑to‑one** with its location—formally, `(i₁,j₁) ≠ (i₂,j₂) ⇒ f_{i₁,j₁} ≠ f_{i₂,j₂}` (Eq. 5, p.4). Thus, tokens carry position **without** any learned positional channel. :contentReference[oaicite:7]{index=7}

**Recommended setting.** Set the wavelet depth to `n = log₂(Ph)` so that the 4ⁿ sub‑bands align exactly with the patch grid; this enables **patch‑wise spectral pooling** (p.4). :contentReference[oaicite:8]{index=8}

---

## Results

**ImageNet‑100** (*scaling study, Table 3*) :contentReference[oaicite:9]{index=9}

| Backbone | Params (M) | GFLOPs | Top‑1 ↑ |
|---|---:|---:|---:|
| **ViM‑T** | 6.42 → 6.65 | 0.16 → 0.21 | **75.10 → 80.41** (+5.31) |
| **ViM‑S** | 23.91 → 24.58 | 0.33 → 0.47 | **79.12 → 84.50** (+5.38) |
| **ViM‑B** | 92.05 → 94.28 | 0.65 → 1.10 | **83.60 → 86.33** (+2.73) |
| **ViT‑T** | 5.72 → 6.06 | 0.91 → 0.98 | **67.64 → 76.04** (+8.40) |
| **ViT‑S** | 22.05 → 23.18 | 3.22 → 3.46 | **74.60 → 80.34** (+5.74) |
| **ViT‑B** | 86.57 → 90.59 | 12.02 → 12.84 | **76.23 → 80.62** (+4.39) |

**Robustness to shuffling** (*Table 4*). LGP nearly eliminates sensitivity to **patch order** and **band order**: ViM‑S drops **−12.66%** when shuffling patches, while **ViM‑S+LGP** drops only **−0.12%**; ViT‑S shows a similar pattern (−4.29% vs. −0.09%). :contentReference[oaicite:10]{index=10}

| Model | Normal | Patch Shuffle | Band Shuffle |
|---|---:|---:|---:|
| ViM‑S | 79.12 | 66.46 (−12.66) | — |
| **ViM‑S + LGP** | **84.50** | **84.38 (−0.12)** | **84.18 (−0.32)** |
| ViT‑S | 74.60 | 70.31 (−4.29) | — |
| **ViT‑S + LGP** | **80.34** | **80.25 (−0.09)** | **80.32 (−0.02)** |

**ImageNet‑1K** (*Table 6*). Out‑of‑the‑box, LGP yields **~+1%** absolute top‑1 at this scale (no extra tuning). :contentReference[oaicite:11]{index=11}

| Backbone | Baseline | +LGP | Gain |
|---|---:|---:|---:|
| ViT‑Small | 75.78 | 76.83 | +1.05 |
| ViM‑Small | 79.52 | 80.91 | +1.39 |
| SwinV2‑Small | 82.71 | 83.61 | +0.90 |
| MambaVision‑Small | 83.01 | 83.86 | +0.85 |

**Breadth.** Across **32 public backbones** spanning six paradigms (plain/distilled ViTs, frequency mixers, local‑window hierarchies, Conv‑Transformer hybrids, efficient/linear attention, state‑space), LGP **consistently boosts** ImageNet‑100 accuracy—e.g., **LinearViT‑Tiny +5.95** and **ViM‑Tiny +5.31**—with modest parameter/compute overheads (Table 5). :contentReference[oaicite:12]{index=12}

---

