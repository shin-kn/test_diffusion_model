# 1D Diffusion Model for Time-Series Anomaly Detection

I read some papers on diffusion models and got interested in using them for anomaly detection,
so I ran some experiments with a 1D U-Net trained in the EDM framework.

I trained the model on normal (OK) samples only, assuming that defective (NG) samples
would be reconstructed with a larger error.

However, the model reconstructed NG samples almost as well as OK ones
(the error was only about 1.05-1.1x larger), so this score did not separate them well.
My hypothesis is that these defects change how often certain patterns appear,
rather than creating new shapes.

I then tried the exact likelihood from the same kind of diffusion model. With the voltage channel
and a larger dataset, it reached an AUC of about 0.77. This is not a like-for-like comparison with
the reconstruction results, which used the current channel and a smaller dataset.

> The dataset, trained weights and result files are not included.

## Method

| Part | What is used |
| --- | --- |
| Network | 1D U-Net: 7 downsampling levels, skip connections, noise-level embedding, about 10M parameters (`src/models/u_net.py`) |
| Framework | EDM preconditioning and loss weighting, with EMA weights for evaluation (`src/models/edm.py`) |
| Solver | Probability-flow ODE with Heun's method |

Two anomaly scores are implemented.

1. **Reconstruction error**: add noise up to `sigma_start`, solve the ODE back to 0, and compare with the input. Only the largest point-wise errors are summed.
2. **Likelihood**: compute `log p(x)` with the instantaneous change-of-variables formula. The divergence is approximated with the Hutchinson estimator and split per point, so each section of the waveform also gets its own score.

## Layout

```
src/models/               U-Net and EDM denoiser
scripts/train_*.py        training (one model per channel)
scripts/test_voltage.py   reconstruction-error score (test_current.py is the same for the other channel)
scripts/test_both.py      combine both channels (train both on the same dataset first)
scripts/test_likelihood.py  likelihood score
scripts/diagnose.py       check whether the model fails to restore OK, or restores NG too well
scripts/user_test.py      plot original vs. reconstructed waveforms
```

## Data format

Put an `.npz` file at `data/processed/signal_all_n10000.npz` (`train_current.py` uses `signal_subset_n10000.npz`).

| Key | Shape | Content |
| --- | --- | --- |
| `x` | (N, 2, L) | waveforms (channel 0: current, channel 1: voltage) |
| `label` | (N,) | 0, 1 = OK / 2, 3 = NG |
| `label_names` | (4,) | display names of the labels |
| `file_name`, `day` | (N,) | sample identifiers |
| `window_s` | (N,) | waveform duration in seconds |

`L` is 10000 in my experiments. It must be a multiple of 10 for `test_likelihood.py` and of 500 for `user_test.py`.

## Usage

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). The scripts run on Apple MPS or CPU (CUDA is not supported yet).

```bash
uv sync
uv run scripts/train_voltage.py
uv run scripts/test_voltage.py
uv run scripts/test_likelihood.py
```

## References

- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ICLR 2021
- Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364), NeurIPS 2022
- Ronneberger et al., [U-Net](https://arxiv.org/abs/1505.04597), MICCAI 2015
