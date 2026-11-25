import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.edm import EDMDenoiser
from models.u_net import UNet1D


plot_first_sample = False
project_root = Path(__file__).resolve().parent.parent
figure_dir = project_root / "data/results/figs"

if plot_first_sample:
    npz_path=project_root / "data/processed/signal_n30000.npz"
    data_set=np.load(npz_path)

    shown=data_set["x"][0][1]
    print(shown[0])

    x=np.linspace(0,data_set["window_s"][0],len(shown))

    plt.plot(x,shown)
    plt.grid(True)
    plt.show()


channel = 0
sigma_start = 2.0
n_steps = 18
sigma_min = 0.002
rho = 7.0
zoom_s = (1.0, 1.1)
error_window = 500
shown_labels = (0, 2, 3)
seed = 0

channel_name = ["current", "voltage"][channel]
weight_names = {0: "data/checkpoints/unet_current_subset_n10000_d7.pt", 1: "data/checkpoints/unet_voltage_all_n10000_d7.pt"}
checkpoint = torch.load(project_root / weight_names[channel], map_location="cpu")
data_set = np.load(project_root / checkpoint.get("npz_name", "data/processed/signal_n30000.npz"))
scale = checkpoint[f"{channel_name}_scale"]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
channel_mults = checkpoint.get("channel_mults", (1, 2, 4, 8, 8))
model = EDMDenoiser(UNet1D(in_channels=1, channel_mults=channel_mults), sigma_data=checkpoint["sigma_data"])
model.load_state_dict(checkpoint["ema_model"])
model = model.to(device).eval()


@torch.no_grad()
def heun_reconstruct(clean, noise):
    ramp = torch.linspace(0, 1, n_steps)
    sigmas = (sigma_start ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_start ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1)]).tolist()
    x = clean + sigma_start * noise
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        d = (x - model(x, torch.full((len(x),), sigma, device=x.device))) / sigma
        x_next = x + (sigma_next - sigma) * d
        if sigma_next > 0:
            d_next = (x_next - model(x_next, torch.full((len(x),), sigma_next, device=x.device))) / sigma_next
            x_next = x + (sigma_next - sigma) * (d + d_next) / 2
        x = x_next
    return x


labels = data_set["label"]
label_names = data_set["label_names"]
train_set = set(checkpoint["train_indices"])
rng = np.random.default_rng(seed)
shown_indices = [
    int(rng.choice([i for i in range(len(labels)) if labels[i] == label and i not in train_set])) for label in shown_labels
]

clean = torch.from_numpy(data_set["x"][shown_indices, channel : channel + 1, :] / scale).float()
noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
recon = heun_reconstruct(clean.to(device), noise.to(device)).cpu()
weight = model.loss_weight(torch.tensor(sigma_start)).item()
local_errors = (weight * (recon - clean) ** 2)[:, 0].reshape(len(shown_indices), -1, error_window).mean(dim=2)

unit = ["A", "V"][channel]
plt.rcParams["font.family"] = "Hiragino Sans"
fig, axes = plt.subplots(len(shown_indices), 3, figsize=(16, 3.2 * len(shown_indices)), squeeze=False)
for row, i in enumerate(shown_indices):
    original = clean[row, 0].numpy() * scale
    reconstructed = recon[row, 0].numpy() * scale
    t = np.linspace(0, data_set["window_s"][i], len(original))
    in_zoom = (t >= zoom_s[0]) & (t < zoom_s[1])
    error_t = (np.arange(local_errors.shape[1]) + 0.5) * error_window * (t[1] - t[0])
    name = f"{label_names[labels[i]]} / {data_set['file_name'][i]} / 誤差 {local_errors[row].mean():.4f}"

    axes[row, 0].plot(t, original, lw=0.3, label="元")
    axes[row, 0].plot(t, reconstructed, lw=0.3, label="再構成")
    axes[row, 0].set_title(name, fontsize=9)
    axes[row, 1].plot(t[in_zoom], original[in_zoom], label="元")
    axes[row, 1].plot(t[in_zoom], reconstructed[in_zoom], label="再構成")
    axes[row, 1].set_title(f"拡大 {zoom_s[0]}–{zoom_s[1]} s", fontsize=9)
    axes[row, 2].plot(error_t, local_errors[row].numpy(), marker=".")
    axes[row, 2].set_title(f"{error_window} 点ごとの誤差", fontsize=9)
    axes[row, 0].set_ylabel(unit)
    axes[row, 1].legend(fontsize=8)
    for ax in axes[row]:
        ax.grid(True)
for ax in axes[-1]:
    ax.set_xlabel("s")
plt.tight_layout()
figure_dir.mkdir(parents=True, exist_ok=True)
figure_path = figure_dir / f"reconstruction_{channel_name}_sigma{sigma_start:g}.png"
fig.savefig(figure_path, dpi=150)
print(f"saved -> {figure_path}")

print("\n| ラベル | ファイル | 再構成誤差 λ(σ)(x̂−x)² の平均 | 元と再構成の相関 |")
print("| --- | --- | --- | --- |")
for row, i in enumerate(shown_indices):
    corr = np.corrcoef(clean[row, 0].numpy(), recon[row, 0].numpy())[0, 1]
    print(f"| {label_names[labels[i]]} | {data_set['file_name'][i]} | {local_errors[row].mean():.3f} | {corr:.3f} |")
