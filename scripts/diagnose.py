from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.edm import EDMDenoiser
from models.u_net import UNet1D

sigma_starts = (0.2, 0.5, 1.0, 2.0, 5.0)
n_steps = 18
sigma_min = 0.002
rho = 7.0
n_repeats = 2
batch_size = 16
n_train_ok = 60
point_error_quantile = 0.99
seed = 0

project_root = Path(__file__).resolve().parent.parent
weight_path = project_root / "data/checkpoints/unet_current_subset_n10000_d7.pt"
result_path = project_root / f"data/results/diagnose_{weight_path.stem}.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(weight_path, map_location="cpu")
channel_name = "voltage" if "voltage_scale" in checkpoint else "current"
channel = {"current": 0, "voltage": 1}[channel_name]
scale = checkpoint[f"{channel_name}_scale"]
channel_mults = checkpoint.get("channel_mults", (1, 2, 4, 8, 8))
model = EDMDenoiser(UNet1D(in_channels=1, channel_mults=channel_mults), sigma_data=checkpoint["sigma_data"])
model.load_state_dict(checkpoint["ema_model"])
model = model.to(device).eval()

npz = np.load(project_root / checkpoint.get("npz_name", "data/processed/signal_n30000.npz"))
labels = npz["label"]
train_set = set(checkpoint["train_indices"])
rng = np.random.default_rng(seed)
groups = {
    "①学習OK": sorted(rng.choice(sorted(train_set), size=min(n_train_ok, len(train_set)), replace=False).tolist()),
    "②評価OK": [i for i in range(len(labels)) if labels[i] <= 1 and i not in train_set],
    "③NG-A": [i for i in range(len(labels)) if labels[i] == 2],
    "④NG-B": [i for i in range(len(labels)) if labels[i] == 3],
}
indices = [i for group in groups.values() for i in group]
group_of = np.array([name for name, group in groups.items() for _ in group])
signal = torch.from_numpy(npz["x"][indices, channel : channel + 1, :] / scale).float()
is_eval = group_of != "①学習OK"
print(f"{weight_path.name} ({channel_name}) / " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()) + f" / device {device}")


def edm_sigmas(sigma_max):
    ramp = torch.linspace(0, 1, n_steps)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sigmas, torch.zeros(1)]).tolist()


@torch.no_grad()
def heun_reconstruct(clean, noise, sigma_start):
    x = clean + sigma_start * noise
    sigmas = edm_sigmas(sigma_start)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        d = (x - model(x, torch.full((len(x),), sigma, device=x.device))) / sigma
        x_next = x + (sigma_next - sigma) * d
        if sigma_next > 0:
            d_next = (x_next - model(x_next, torch.full((len(x),), sigma_next, device=x.device))) / sigma_next
            x_next = x + (sigma_next - sigma) * (d + d_next) / 2
        x = x_next
    return x


def auc(scores, positive):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


eval_ok_positions = np.flatnonzero(group_of == "②評価OK")
other_ok = np.array([rng.choice(eval_ok_positions[eval_ok_positions != k]) for k in range(len(indices))])

rows = []
generator = torch.Generator().manual_seed(seed)
for sigma_start in sigma_starts:
    weight = model.loss_weight(torch.tensor(sigma_start)).item()
    point_errors = np.zeros((len(indices), signal.shape[-1]), dtype=np.float32)
    noisy_errors = np.zeros(len(indices))
    for _ in range(n_repeats):
        for batch in tqdm(list(batched(range(len(indices)), batch_size)), desc=f"σ={sigma_start}", leave=False):
            clean = signal[list(batch)]
            noise = torch.randn(clean.shape, generator=generator)
            recon = heun_reconstruct(clean.to(device), noise.to(device), sigma_start).cpu()
            point_errors[list(batch)] += (weight * (recon - clean) ** 2)[:, 0].numpy() / n_repeats
            noisy_errors[list(batch)] += (weight * (sigma_start * noise) ** 2).mean(dim=(1, 2)).numpy() / n_repeats
    other_errors = (weight * (signal[other_ok] - signal) ** 2).mean(dim=(1, 2)).numpy()

    mean_errors = point_errors.mean(axis=1)
    threshold = np.quantile(point_errors[is_eval], point_error_quantile)
    top_errors = np.where(point_errors >= threshold, point_errors, 0).sum(axis=1)

    row = {"sigma_start": sigma_start, "lambda": weight}
    for name in groups:
        mask = group_of == name
        row[f"{name}_mean"] = mean_errors[mask].mean()
        row[f"{name}_top1%"] = top_errors[mask].mean()
    row["基準a_ノイズそのまま"] = noisy_errors[is_eval].mean()
    row["基準b_別のOK"] = other_errors[is_eval].mean()
    ok_mask = group_of == "②評価OK"
    for name in ("③NG-A", "④NG-B"):
        mask = ok_mask | (group_of == name)
        row[f"AUC_{name}_mean"] = auc(mean_errors[mask], group_of[mask] == name)
        row[f"AUC_{name}_top1%"] = auc(top_errors[mask], group_of[mask] == name)
    rows.append(row)
    print(f"σ={sigma_start} done")

result = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4g}".format):
    print("\n[点平均の誤差] (基準 a/b も点平均)")
    print(result[["sigma_start"] + [f"{n}_mean" for n in groups] + ["基準a_ノイズそのまま", "基準b_別のOK"]].to_string(index=False))
    print("\n[上位 1% の点だけ足した誤差]")
    print(result[["sigma_start"] + [f"{n}_top1%" for n in groups]].to_string(index=False))
    print("\n[AUC (②評価OK vs NG)]")
    print(result[["sigma_start"] + [c for c in result.columns if c.startswith("AUC_")]].to_string(index=False))

result_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(result_path, index=False)
print(f"saved -> {result_path}")
