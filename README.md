# DDIM Inpainting on MNIST

This project implements DDIM-based image inpainting on 32×32 MNIST digits.

The task:  
take a digit image, **cut out a central square (hole)**, and train a diffusion model to fill the missing region so it looks like a realistic MNIST digit.

The implementation follows the CSE 573 Project 2 spec and re-implements:

- sinusoidal time embeddings
- residual blocks with time injection
- 2D self-attention
- a time-conditioned UNet backbone
- diffusion buffers (α, β, ᾱ, etc.)
- DDIM sampling with data consistency for inpainting

Everything is implemented in a single notebook:

> `DDIM_Inpainting_Student.ipynb`

---

## Model & Diffusion

- **Backbone:** UNet with:
  - base channels `base = 128`
  - time embedding dimension `emb = 512`
  - residual blocks with time injection
  - GroupNorm + SiLU activations
  - mid-layer self-attention (8×8)
- **Conditioning channels:**
  - `x_t` – noisy image at time t
  - `m` – binary mask (1 = known pixels, 0 = hole)
  - `y` – observed image with hole
  - optional self-conditioning `x0_sc`
  - optional coordinate channels `(x, y)` (coord-conv)

- **Diffusion:**
  - steps: `T = 400`
  - schedule: cosine ᾱ(t)
  - prediction space: **v-prediction** (`pred="v"`)
  - DDIM sampling with `eta = 0.0` (deterministic)

---

## Training & Final Config

Final training call (used for the results in this repo):

```python
train(
    epochs=26,
    batch_size=128,
    lr=2e-4,

    steps=400,
    beta_start=1e-4,
    beta_end=2e-2,
    beta_schedule="cosine",

    base=128,
    emb=512,

    sample_every=1600,
    sample_steps=100,

    center_box=12,

    dc_repeats=3,
    dc_fixed_z=True,

    pred="v",
    p2_k=1.0,
    p2_gamma=1.0,

    hole_weight=8.0,

    seed=1234,
    clip_grad=1.0,

    ema_decay=0.999,
    warmup_steps=500,

    self_cond=True,
    eta=0.0,
    coord_conv=True,

    grad_accum=1,
)
```

For more details about environment and exact run commands, see: 
```text
Run.md
```

## Results

```json
{
  "psnr_hole": 14.41,
  "l1_hole": 0.201
}
```

Some qualitative results are in results/:

panel_final.png – input / masked / inpainted
samples.png – grid of inpainted digits

## How to Run

1. Install dependencies (PyTorch + torchvision, etc.).
2. Open DDIM_Inpainting.ipynb in Jupyter or Google Colab (GPU recommended).
3. Run all cells top to bottom to:
  - train the model
  - generate inpainting samples
  - compute validation metrics.

See Run.md for exact machine specs and the precise training call.
