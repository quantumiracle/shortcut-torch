# Shortcut-Torch

## A PyTorch Implementation of the Shortcut Generative Model

This repository provides a simple implementation of the Shortcut generative model in PyTorch. The implementation follows flow matching principles rather than DDPM (Denoising Diffusion Probabilistic Models).

Unofficial implementation for [One Step Diffusion via Shortcut Models
](https://arxiv.org/abs/2410.12557)

## Overview

The Shortcut model extends traditional generative models by incorporating concepts from progressive distillation (for diffusion models) and reflow techniques (for flow matching) during training. This approach can lead to faster sampling and improved efficiency compared to traditional approaches.

## Features

- PyTorch implementation of the Shortcut model
- Flow matching-based approach
- MNIST dataset for quick testing and demonstration
- Optional implementation of techniques from Faster-DiT (not fully tested)

## Installation

```bash
# Clone the repository
git clone https://github.com/quantumiracle/shortcut-torch.git
cd shortcut-torch

# Install dependencies (TBD)
# pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train_shortcut.py
```

### Results

20 epochs (<3 hours) on MINST:

128 steps:

![Alt text](https://github.com/quantumiracle/shortcut-torch/blob/master/results/ep20_w2.0_steps128.gif)

4 steps:

![Alt text](https://github.com/quantumiracle/shortcut-torch/blob/master/results/ep20_w2.0_steps4.gif)

2 steps:

![Alt text](https://github.com/quantumiracle/shortcut-torch/blob/master/results/ep20_w2.0_steps2.gif)

1 step:

![Alt text](https://github.com/quantumiracle/shortcut-torch/blob/master/results/ep20_w2.0_steps1.gif)


## Takeaways
- Flow matching learns faster and better than standard DDPM
- 1 or 2 step inference with Shortcut do not visually looks good (to be tuned)
- For MNIST class condition, one-hot encoding is critical
- For MNIST images (28x28), patch size 1 is critical, patch size 2 performs significantly worse
- Faster-DiT techniques do not significantly improve for this simple setting (I do not carefully tune the configurations)
The implementation is based on flow matching principles and includes optional techniques from Faster-DiT. While these techniques are implemented, they have not been extensively tested and may not significantly improve performance in this specific implementation.


## Learning Dynamics

For an in-depth explanation of the Shortcut model's learning dynamics, refer to [this excellent blog post](https://www.tedstaley.com/posts/short/shortcut.html) by Ted Staley.

## References
The code is refactored from the two good references repos:
- [Official JAX Implementation](https://github.com/kvfrans/shortcut-models)
- [Unofficial PyTorch Implementation](https://github.com/smileyenot983/shortcut_pytorch)

## License

[Your license information here]

## Citation

If you use this code in your research, please cite:

```
[Citation information]
```

