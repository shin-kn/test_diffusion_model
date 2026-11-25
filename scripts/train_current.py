from itertools import batched
from pathlib import Path
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from models.edm import EDMDenoiser, sample_training_sigma
from models.u_net import UNet1D

iter_loop=60
batch_size=30
sigma_data = 0.5
ema_decay = 0.999
p_mean = -0.4
channel_mults = (1, 2, 4, 8, 8, 8, 8, 8)

project_root = Path(__file__).resolve().parent.parent
npz_name = "data/processed/signal_subset_n10000.npz"
npz_path = project_root / npz_name
weight_path = project_root / "data/checkpoints/unet_current_subset_n10000_d7.pt"

current_scale = 130.0
train_labels = (0, 1)
holdout_ratio = 0.2
seed = 0

random.seed(seed)
torch.manual_seed(seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

npz = np.load(npz_path)
ok_indices = [i for i, label in enumerate(npz["label"]) if label in train_labels]
random.shuffle(ok_indices)
n_holdout = int(len(ok_indices) * holdout_ratio)
train_indices = sorted(ok_indices[n_holdout:])

current = npz["x"][train_indices, 0:1, :] / current_scale
data_set=torch.from_numpy(current).float().to(device)
print(f"train {len(data_set)} / holdout OK {n_holdout} / device {device}")

model=EDMDenoiser(UNet1D(in_channels=1, channel_mults=channel_mults), sigma_data=sigma_data).to(device)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

optimizer=torch.optim.Adam(model.parameters(), lr=2e-4)

def loss_func(pred, target, weight):
    return (weight[:, None, None] * (pred - target) ** 2).mean()


@torch.no_grad()
def update_ema(step):
    decay = min(ema_decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)


step = 0
n_batches = -(-len(data_set) // batch_size)
for l in tqdm(range(iter_loop), desc="epoch"):
    order = list(range(len(data_set)))
    random.shuffle(order)
    loss_sum, n_samples = 0.0, 0
    progress = tqdm(batched(order,batch_size), total=n_batches, desc=f"epoch {l}", leave=False)
    for batch in progress:
        clean = data_set[list(batch)]
        noise_level=sample_training_sigma(len(batch), device, p_mean=p_mean)
        noisy = clean + noise_level[:, None, None] * torch.randn_like(clean)

        pred=model(noisy, noise_level)
        loss = loss_func(pred, clean, model.loss_weight(noise_level))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(step)
        step += 1

        loss_sum += loss.item() * len(batch)
        n_samples += len(batch)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    tqdm.write(f"epoch {l}: loss {loss_sum / n_samples:.4f}")


weight_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "current_scale": current_scale,
        "sigma_data": sigma_data,
        "p_mean": p_mean,
        "channel_mults": channel_mults,
        "train_indices": train_indices,
        "npz_name": npz_name,
    },
    weight_path,
)
print(f"saved -> {weight_path}")
