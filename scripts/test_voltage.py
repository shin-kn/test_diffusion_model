from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.edm import EDMDenoiser
from models.u_net import UNet1D

sigma_start = 2.0
n_steps = 18
sigma_min = 0.002
rho = 7.0
n_repeats = 4
batch_size = 16
anomaly_rate = None
point_error_quantile = 0.99
seed = 0

project_root = Path(__file__).resolve().parent.parent
weight_path = project_root / "data/checkpoints/unet_voltage_all_n10000_d7.pt"
result_path = project_root / "data/results/test_voltage_all_n10000_d7_s2_errors.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(weight_path, map_location="cpu")
channel_mults = checkpoint.get("channel_mults", (1, 2, 4, 8, 8))
model = EDMDenoiser(UNet1D(in_channels=1, channel_mults=channel_mults), sigma_data=checkpoint["sigma_data"])
model.load_state_dict(checkpoint["ema_model"])
model = model.to(device).eval()
voltage_scale = checkpoint["voltage_scale"]

npz = np.load(project_root / checkpoint.get("npz_name", "data/processed/signal_n30000.npz"))
labels = npz["label"]
label_names = npz["label_names"]
train_set = set(checkpoint["train_indices"])
test_indices = [i for i in range(len(labels)) if i not in train_set]
is_ng = labels[test_indices] >= 2
voltage = torch.from_numpy(npz["x"][test_indices, 1:2, :] / voltage_scale).float()
print(f"test {len(test_indices)} (OK {(~is_ng).sum()}, NG {is_ng.sum()}) / device {device}")


def point_loss_func(pred, target, weight):
    return (weight[:, None, None] * (pred - target) ** 2)[:, 0]


def edm_sigmas(sigma_max):
    ramp = torch.linspace(0, 1, n_steps)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sigmas, torch.zeros(1)]).tolist()


@torch.no_grad()
def heun_reconstruct(clean, noise):
    x = clean + sigma_start * noise
    sigmas = edm_sigmas(sigma_start)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        s = torch.full((len(x),), sigma, device=x.device)
        d = (x - model(x, s)) / sigma
        x_next = x + (sigma_next - sigma) * d
        if sigma_next > 0:
            s_next = torch.full((len(x),), sigma_next, device=x.device)
            d_next = (x_next - model(x_next, s_next)) / sigma_next
            x_next = x + (sigma_next - sigma) * (d + d_next) / 2
        x = x_next
    return x


generator = torch.Generator().manual_seed(seed)
weight = model.loss_weight(torch.tensor([sigma_start]))
point_errors = np.zeros((len(test_indices), voltage.shape[-1]))
for _ in tqdm(range(n_repeats), desc="repeat"):
    for batch in tqdm(list(batched(range(len(test_indices)), batch_size)), desc="batch", leave=False):
        clean = voltage[list(batch)]
        noise = torch.randn(clean.shape, generator=generator)
        recon = heun_reconstruct(clean.to(device), noise.to(device)).cpu()
        point_errors[list(batch)] += point_loss_func(recon, clean, weight.expand(len(batch))).numpy() / n_repeats

point_threshold = np.quantile(point_errors, point_error_quantile)
errors = np.where(point_errors >= point_threshold, point_errors, 0).sum(axis=1)
errors_mean = point_errors.mean(axis=1)
print(f"点ごとの閾値 {point_threshold:.4f} (上位 {1 - point_error_quantile:.1%}), 閾値を超えた点の数の中央値 {np.median((point_errors >= point_threshold).sum(axis=1)):.0f}")


def auc(scores, positive):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


test_labels = labels[test_indices]


def print_aucs():
    ok_mask = test_labels <= 1
    print(f"  AUC (全NG)   {auc(errors, is_ng):.3f}")
    for label in (2, 3):
        mask = ok_mask | (test_labels == label)
        print(f"  AUC ({label_names[label]}) {auc(errors[mask], test_labels[mask] == label):.3f}")


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
    print_aucs()


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

print("\n[比較: 全点の平均 (閾値なし)]")
print(f"  AUC (全NG)   {auc(errors_mean, is_ng):.3f}")
print(f"\n正答率の基準 (全部 OK と答えた場合) {(~is_ng).mean():.3f}")

result_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "file_name": npz["file_name"][test_indices],
        "day": npz["day"][test_indices],
        "label": label_names[test_labels],
        "error": errors,
        "error_mean": errors_mean,
    }
).sort_values("error", ascending=False).to_csv(result_path, index=False)
print(f"saved -> {result_path}")
