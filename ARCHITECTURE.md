# Architecture: Dual-Head Adversarial U-Net for Skin Burn Analysis

## 1. Overview

This document specifies the architecture for a **Dual-Head Adversarial U-Net**, a segmentation
model designed to localize burn boundaries on skin images while actively suppressing its
ability to encode **Fitzpatrick skin tone** in its learned features. The goal is a segmentation
model whose bottleneck representation is predictive of burn geometry but *not* predictive of
skin tone, so that segmentation accuracy does not systematically vary across skin tone groups.

The model has a single shared encoder feeding two heads that are trained with opposing
objectives:

- **Path A (Segmentation Decoder):** a standard U-Net decoder with skip connections that
  upsamples bottleneck features back to full resolution and emits raw per-pixel logits
  over burn regions (paired with `BCEWithLogitsLoss`; sigmoid is applied only at
  inference time to obtain the probability mask).
- **Path B (Adversarial Fitzpatrick Classifier):** a lightweight classification head that
  receives the *same* bottleneck features through a **Gradient Reversal Layer (GRL)**, and
  predicts the Fitzpatrick skin-tone class (I–VI). Because gradients from this head are
  negated before reaching the encoder, training the classifier well simultaneously trains
  the encoder to produce features from which skin tone *cannot* be recovered.

This is a standard domain-adversarial setup (Ganin & Lempitsky, "Domain-Adversarial Training
of Neural Networks") repurposed for fairness/debiasing rather than domain adaptation: here,
"domain" = Fitzpatrick skin tone class, and we want the encoder invariant to it.

```
                        ┌─────────────────────┐
                        │   Input Image (RGB)  │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   Shared Encoder      │
                        │ (U-Net / ResNet-34/50)│
                        │  produces skip feats  │
                        │  + bottleneck feat z  │
                        └──────────┬───────────┘
                                   │  z (bottleneck features)
                     ┌─────────────┴─────────────┐
                     │                           │
          ┌──────────▼──────────┐     ┌──────────▼──────────┐
          │   Path A: Decoder    │     │  Gradient Reversal   │
          │  (skip connections   │     │       Layer (GRL)    │
          │   from encoder)      │     │  forward: identity   │
          │                      │     │  backward: −λ · grad │
          └──────────┬──────────┘     └──────────┬──────────┘
                     │                           │
          ┌──────────▼──────────┐     ┌──────────▼──────────┐
          │      Conv 1x1        │     │  Path B: Classifier   │
          │  → Burn mask logits  │     │  (GAP + MLP)          │
          │  shape: [B,1,H,W]    │     │  → logits ∈ R^6       │
          │  (sigmoid at         │     │  (Fitzpatrick I–VI)   │
          │   inference only)    │     │                       │
          └──────────────────────┘     └───────────────────────┘
```

## 2. Component Breakdown

### 2.1 Shared Encoder

- Backbone: standard U-Net contracting path, **or** a ResNet-34/ResNet-50 encoder
  (ImageNet-pretrained) used U-Net style — configurable via a `backbone` argument.
- Produces a feature pyramid: intermediate activations at each downsampling stage are
  retained as **skip connections** for Path A, and the final, most-downsampled feature map
  is the **bottleneck representation `z`** shared by both heads.
- This encoder is the single point of "debiasing pressure": both heads read from `z`
  (and skips, for Path A), and only the encoder's weights are pushed in a direction that
  simultaneously helps segmentation and hurts skin-tone classification.

### 2.2 Path A — Segmentation Decoder (primary task)

- Standard U-Net expansive path: transposed convolutions (or upsample + conv) at each
  stage, concatenating the corresponding encoder skip connection at matching resolution.
- Ends in a `1x1` convolution collapsing channels to 1, producing raw **logits** at the
  original input resolution — no sigmoid is applied inside the model. Sigmoid is applied
  only when a probability mask is needed (e.g. at inference/visualization time), keeping
  the loss numerically stable.
- Loss: `BCEWithLogitsLoss` (+ optionally Dice, computed on `sigmoid(logits)`) against
  ground-truth burn boundary masks. This is the task loss, `L_seg`.

### 2.3 Path B — Adversarial Fitzpatrick Classification Head (debiasing task)

- Takes the bottleneck feature map `z` (not the decoder output — the point is to purge
  skin-tone signal from the *shared* representation, not from the mask).
- Passes `z` through the **Gradient Reversal Layer** (see Section 3) before any
  classifier weights.
- Head architecture: Global Average Pooling → small MLP (e.g. `Linear → ReLU → Dropout →
  Linear`) → logits over 6 classes (Fitzpatrick types I through VI).
- Loss: standard multi-class cross-entropy against Fitzpatrick label, `L_adv`.

### 2.4 Combined Objective

```
L_total = L_seg  +  β · L_adv
```

- `L_seg`: segmentation loss (`BCEWithLogitsLoss`, optionally + Dice) on Path A's raw
  logit output — minimized normally.
- `L_adv`: classification loss on Path B output — the *classifier head's* weights are
  minimized on this loss (get better at predicting skin tone), but because of the GRL,
  the *encoder's* weights are effectively pushed to **maximize** `L_adv` (get worse at
  letting skin tone be predicted).
- `β`: a scalar weighting the adversarial loss contribution relative to segmentation.
- This single combined loss with GRL is what implements the minimax game without needing
  a manually alternated training loop — the sign flip happens inside the backward pass.

## 3. The Gradient Reversal Layer (GRL): Mathematical Objective

The GRL is a custom autograd operation, `R_λ`, defined identically on the forward pass but
inverted on the backward pass:

**Forward pass** (identity function):

```
R_λ(x) = x
```

**Backward pass** (negated, scaled gradient):

```
∂R_λ/∂x = −λ · I
```

i.e., during backpropagation, the incoming gradient `∂L_adv/∂R_λ(x)` is multiplied by
`−λ` before being propagated further back into the encoder. `λ ≥ 0` is a tunable
adversarial strength coefficient (often annealed from 0 → 1 over training, per the
Ganin & Lempitsky schedule, so the encoder is not destabilized by strong adversarial
gradients early in training).

### Why this implements a minimax game

Let `E` denote encoder parameters, `C` denote classifier-head parameters, and
`z = f_E(x)` the bottleneck features. Without the GRL, joint gradient descent on
`L_adv` over `[E, C]` would do:

```
E ← E − η · ∂L_adv/∂E        (encoder learns to HELP classify skin tone)
C ← C − η · ∂L_adv/∂C        (classifier learns to classify skin tone)
```

This is the opposite of what we want for the encoder. The GRL sits between `z` and the
classifier head, so the chain rule through it flips the sign of only the term flowing into
`E`:

```
∂L_adv/∂E  (as seen by optimizer)  =  −λ · (true ∂L_adv/∂E)
∂L_adv/∂C  (as seen by optimizer)  =  +1 · (true ∂L_adv/∂C)    (unaffected)
```

A single unmodified SGD/Adam step over the whole graph then performs, simultaneously:

```
min_C  L_adv(C; E)              — classifier tries to succeed
max_E  L_adv(C; E)  (via −λ)     — encoder tries to make it fail
```

subject to `E` also minimizing `L_seg` at the same time (Path A's gradients into `E` flow
normally, unreversed). At convergence, the encoder is pushed toward a saddle point where
segmentation performance is maximized while the best possible Fitzpatrick classifier
trained on `z` still performs near chance — i.e., `z` becomes approximately invariant to
skin tone while remaining informative for burn boundaries. This is the same minimax
formulation as Domain-Adversarial Neural Networks (DANN), with "domain" replaced by
"Fitzpatrick class."

## 4. File Structure

```
skinburn_analyser/
├── ARCHITECTURE.md          # this document
├── model/
│   ├── __init__.py
│   ├── layers.py            # GradientReversalLayer (custom autograd.Function + nn.Module wrapper),
│   │                         # ConvBlock, UpBlock, and other reusable building blocks
│   ├── encoder.py            # Shared encoder: U-Net contracting path or ResNet-34/50 backbone,
│   │                         # exposes skip connections + bottleneck feature map
│   ├── decoder.py            # Path A: U-Net expansive path consuming skips + bottleneck,
│   │                         # ends in 1x1 conv → burn mask logits (no sigmoid; paired
│   │                         # with BCEWithLogitsLoss, sigmoid applied at inference only)
│   ├── adversarial_head.py   # Path B: GRL → GAP → MLP → Fitzpatrick logits
│   └── model.py              # DualHeadAdversarialUNet: wires encoder + decoder + adversarial
│                              # head together, exposes forward() returning (mask, tone_logits)
├── data/
│   ├── __init__.py
│   └── dataset.py            # BurnDataset + get_dataloaders(): CSV-driven 80/10/10 split,
│                              # in-place random hflip augmentation (train split only)
├── training/
│   ├── __init__.py
│   ├── losses.py             # BCEWithLogitsLoss (+ optional Dice) for L_seg, CrossEntropy
│   │                         # for L_adv, combined L_total
│   ├── lambda_schedule.py    # GRL λ annealing schedule (e.g. sigmoid ramp per DANN paper)
│   └── train.py               # Training loop, optimizer setup, logging, checkpointing
├── configs/
│   └── default.yaml           # Hyperparameters: backbone choice, β, λ schedule, LR, etc.
└── evaluation/
    ├── __init__.py
    ├── segmentation_metrics.py  # IoU, Dice score for burn mask quality
    └── fairness_metrics.py      # Per-Fitzpatrick-class segmentation performance gap,
                                   # adversarial classifier accuracy (should trend to chance)
```

## 5. Evaluation Strategy (for context, not implementation yet)

Two axes must be reported, not just one:

1. **Segmentation quality**: Dice / IoU on burn masks, both overall and **broken out per
   Fitzpatrick class**, to confirm the debiasing objective is actually closing performance
   gaps rather than just degrading average accuracy.
2. **Debiasing effectiveness**: accuracy of the adversarial classifier head at
   convergence. Success looks like this accuracy approaching the chance rate for 6 classes
   (~16.7%), indicating `z` no longer linearly encodes skin tone — *not* zero, since some
   residual correlation between tone and legitimate skin-texture cues may be unavoidable.

## 6. Open Design Parameters (to decide before implementation)

- Backbone choice: pure U-Net encoder vs. pretrained ResNet-34/50 (affects `encoder.py`
  interface and skip-connection channel counts).
- `β` (adversarial loss weight) and the `λ` annealing schedule — both need tuning and
  likely a short ablation.
- Whether Fitzpatrick labels are ground-truth annotated or predicted/pseudo-labeled
  upstream (affects `dataset.py` assumptions).
- Dropout / capacity of the adversarial MLP head — too weak a classifier gives a weak,
  uninformative adversarial signal.
