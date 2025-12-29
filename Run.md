# Run.md

## Machine Specs

**Environment**

- Platform: Google Colab (GPU runtime)
- OS: Linux (Ubuntu-based Colab VM)
- CPU: Colab VM multi-core CPU (default)
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: ~12 GB system RAM

**Software**

- Python: 3.10+ (Colab default at time of run)
- PyTorch: 2.x (preinstalled in Colab, CUDA-enabled build)
- CUDA: 11.x (whatever is bundled with the Colab PyTorch build)
- Other Python packages:
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `tqdm`
  - standard library modules: `math`, `json`, `random`

No custom CUDA/C++ extensions were used.

---

## Exact Commands

All commands are run inside:

> `DDIM_Inpainting.ipynb`

in order, from top to bottom.

### 1. Training

The final submitted model is trained with the following call in the **Training** cell:

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
