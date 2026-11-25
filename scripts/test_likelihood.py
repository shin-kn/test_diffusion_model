import math
from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.edm import EDMDenoiser
from models.u_net import UNet1D

channel = 1
sigma_min = 0.002
sigma_max = 80.0
n_steps = 40
rho = 7.0
n_hutchinson = 2
n_areas = 10
batch_size = 8
anomaly_rate = None
n_eval = 400
seed = 0

project_root = Path(__file__).resolve().parent.parent
channel_name = ["current", "voltage"][channel]
run_name = f"{channel_name}_all_n10000_d7"
weight_path = project_root / f"data/checkpoints/unet_{run_name}.pt"
result_path = project_root / f"data/results/test_likelihood_{run_name}.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(weight_path, map_location="cpu")
channel_mults = checkpoint.get("channel_mults", (1, 2, 4, 8, 8))
model = EDMDenoiser(UNet1D(in_channels=1, channel_mults=channel_mults), sigma_data=checkpoint["sigma_data"])
model.load_state_dict(checkpoint["ema_model"])
model = model.to(device).eval().requires_grad_(False)
scale = checkpoint[f"{channel_name}_scale"]

npz = np.load(project_root / checkpoint.get("npz_name", "data/processed/signal_n30000.npz"))
labels = npz["label"]
label_names = npz["label_names"]
train_set = set(checkpoint["train_indices"])
test_indices = [i for i in range(len(labels)) if i not in train_set]
if n_eval is not None and n_eval < len(test_indices):
    test_indices = sorted(np.random.default_rng(seed).choice(test_indices, n_eval, replace=False).tolist())
test_labels = labels[test_indices]
is_ng = test_labels >= 2
signal = torch.from_numpy(npz["x"][test_indices, channel : channel + 1, :] / scale).float()
print(f"{channel_name}: test {len(test_indices)} (OK {(~is_ng).sum()}, NG {is_ng.sum()}) / device {device}")


def edm_sigmas_ascending():
    ramp = torch.linspace(0, 1, n_steps + 1)
    return (sigma_min ** (1 / rho) + ramp * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho


def velocity_and_divergence(x, sigma, eps):
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = (x - model(x, torch.full((len(x),), sigma, device=x.device))) / sigma
        (vjp,) = torch.autograd.grad((v * eps).sum(), x)
    return v.detach(), (eps * vjp)[:, 0]


def point_log_likelihood(clean, eps):
    sigmas = edm_sigmas_ascending().tolist()
    x = clean
    divergence_integral = torch.zeros(clean.shape[0], clean.shape[-1], device=clean.device)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        h = sigma_next - sigma
        v, div = velocity_and_divergence(x, sigma, eps)
        x_euler = x + h * v
        v_next, div_next = velocity_and_divergence(x_euler, sigma_next, eps)
        x = x + h * (v + v_next) / 2
        divergence_integral += h * (div + div_next) / 2
    prior = -0.5 * (x[:, 0] / sigma_max) ** 2 - math.log(sigma_max) - 0.5 * math.log(2 * math.pi)
    return prior + divergence_integral


generator = torch.Generator().manual_seed(seed)
point_log_p = np.zeros((len(test_indices), signal.shape[-1]))
for _ in tqdm(range(n_hutchinson), desc="hutchinson"):
    for batch in tqdm(list(batched(range(len(test_indices)), batch_size)), desc="batch", leave=False):
        clean = signal[list(batch)]
        eps = (torch.randint(0, 2, clean.shape, generator=generator) * 2 - 1).float()
        point_log_p[list(batch)] += point_log_likelihood(clean.to(device), eps.to(device)).cpu().numpy() / n_hutchinson

nll = -point_log_p.mean(axis=1) / math.log(2)
area_nll = -point_log_p.reshape(len(test_indices), n_areas, -1).mean(axis=2) / math.log(2)
errors = nll


def auc(scores, positive):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def print_aucs(scores, indent="  "):
    ok_mask = test_labels <= 1
    values = [auc(scores, is_ng)]
    for label in (2, 3):
        mask = ok_mask | (test_labels == label)
        values.append(auc(scores[mask], test_labels[mask] == label))
    print(f"{indent}AUC 全NG {values[0]:.3f} / {label_names[2]} {values[1]:.3f} / {label_names[3]} {values[2]:.3f}")


def report(name, predicted_ng, threshold):
    tp = (predicted_ng & is_ng).sum()
    fp = (predicted_ng & ~is_ng).sum()
    fn = (~predicted_ng & is_ng).sum()
    tn = (~predicted_ng & ~is_ng).sum()
    print(f"\n[{name}] threshold {threshold:.5f}, NG と判定 {predicted_ng.sum()} 件")
    print(f"  accuracy  {(tp + tn) / len(is_ng):.3f}")
    print(f"  precision {tp / max(tp + fp, 1):.3f} / recall {tp / max(tp + fn, 1):.3f}")
    print(f"  {'':10}予測NG 予測OK\n  真値NG    {tp:6} {fn:6}\n  真値OK    {fp:6} {tn:6}")
    for label in (2, 3):
        mask = test_labels == label
        print(f"  {label_names[label]}: {(predicted_ng & mask).sum()}/{mask.sum()} 検出")
    print_aucs(errors)


print(f"\nNLL [bits/dim]: OK 中央値 {np.median(nll[~is_ng]):.4f} / NG 中央値 {np.median(nll[is_ng]):.4f}")
print("\n[スコアの種類ごとの AUC]")
print("全体の NLL:")
print_aucs(nll)
print("エリアごとの NLL の最大:")
print_aucs(area_nll.max(axis=1))
for area in range(n_areas):
    print(f"エリア {area}:")
    print_aucs(area_nll[:, area])

order = np.argsort(-errors)
sorted_errors = errors[order]
sorted_is_ng = is_ng[order]

n_correct = (~is_ng).sum() + np.concatenate([[0], np.cumsum(np.where(sorted_is_ng, 1, -1))])
best_k = int(np.argmax(n_correct))
best_threshold = sorted_errors[best_k - 1] if best_k > 0 else np.inf
report("正答率が最大になる位置で切る", errors >= best_threshold, best_threshold)

rate = is_ng.mean() if anomaly_rate is None else anomaly_rate
rate_k = int(round(len(errors) * rate))
rate_threshold = sorted_errors[rate_k - 1] if rate_k > 0 else np.inf
report(f"上位 {rate:.1%} を NG とする", errors >= rate_threshold, rate_threshold)

print(f"\n正答率の基準 (全部 OK と答えた場合) {(~is_ng).mean():.3f}")

result_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "file_name": npz["file_name"][test_indices],
        "day": npz["day"][test_indices],
        "label": label_names[test_labels],
        "nll": nll,
        **{f"nll_area{area}": area_nll[:, area] for area in range(n_areas)},
    }
).sort_values("nll", ascending=False).to_csv(result_path, index=False)
print(f"saved -> {result_path}")
