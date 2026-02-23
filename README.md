# CFG Diffusion on MNIST

Classifier-Free Guidance (CFG) Diffusion Model implemented in Julia using Flux.

## Overview

This project implements a **Classifier-Free Guidance (CFG)** denoising diffusion probabilistic model (DDPM) for conditional image generation on the MNIST dataset.

### Key Features

- **Conditional Generation**: Generate specific digits (0-9) by conditioning on class labels
- **Classifier-Free Guidance**: Use guidance scale to control the strength of conditioning
  - Guidance formula: `ε(x_t, c) = ε(x_t) + s(ε(x_t|c) - ε(x_t))`
  - `s=1.0`: Standard conditioning
  - `s>1.0`: Amplified guidance (sharper, more class-consistent samples)
- **U-Net Architecture**: Residual blocks with time and class embeddings
- **MNIST Dataset**: Handwritten digit generation (28×28 grayscale)

## Architecture

### Conditional U-Net
```
Input (28×28×1) ──→ Conv ──→ [ResBlock + TimeEmbed] × N ──→ Bottleneck ──→ [ResBlock + Skip] × N ──→ Conv ──→ Output
                                        ↑                                                          ↑
                                   Class Embedding                                            Skip Connections
```

### Time Embedding
- Sinusoidal positional encoding (similar to Transformers)
- Projected through MLP to match channel dimensions

### Classifier-Free Guidance
During training:
- Randomly drop labels with probability `p_uncond = 0.1`
- Train model to handle both conditional and unconditional generation

During sampling:
- Run two forward passes (conditional + unconditional)
- Combine predictions using guidance scale

## Installation

```bash
git clone https://github.com/kowyo-bot/CFGDiffusion.jl.git
cd CFGDiffusion.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

### Training

```bash
julia --project=. train_mnist.jl
```

Options (edit `train_mnist.jl`):
- `epochs=50`: Number of training epochs
- `batch_size=128`: Batch size
- `lr=2e-4`: Learning rate
- `device=cpu/gpu`: Computation device

### Sampling

The training script automatically generates samples every 10 epochs with different guidance scales:

```julia
# Generate digit "5" with strong guidance
labels = fill(5, 16)  # 16 samples
samples = sample(ddpm, model, (28, 28, 1, 16), labels; guidance_scale=7.5f0)
```

### Guidance Scale Effects

| Scale | Effect |
|-------|--------|
| 1.0 | Standard conditioning, diverse samples |
| 3.0 | Moderate guidance, better class adherence |
| 7.5 | Strong guidance (used in Stable Diffusion), sharp samples |
| 10.0+ | Very strong guidance, may reduce diversity |

## Results

Training progress and samples are saved to `results/`:
- `epoch_XXX_gsY.Y.png`: Sample grids for different guidance scales
- `loss_curve.png`: Training loss over time
- `best_model.jld2`: Best checkpoint
- `final_model.jld2`: Final checkpoint

## Mathematical Background

### Forward Process (Diffusion)
```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
```

### Reverse Process with CFG
```
ε_θ(x_t, c) = ε_θ(x_t) + s·(ε_θ(x_t|c) - ε_θ(x_t))
```

Where:
- `ε_θ(x_t|c)`: Noise prediction with class label
- `ε_θ(x_t)`: Noise prediction without conditioning (null class)
- `s`: Guidance scale

## References

1. Ho & Salimans (2022). ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598)
2. Dhariwal & Nichol (2021). ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233)
3. Ho et al. (2020). ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
4. [Lior Sinai Blog](https://liorsinai.github.io/machine-learning/2023/01/04/denoising-diffusion-3-guidance.html) - Julia implementation reference

## License

MIT License
