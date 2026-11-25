from pathlib import Path

import numpy as np
import pandas as pd

anomaly_rate = None

project_root = Path(__file__).resolve().parent.parent
voltage_result_path = project_root / "data/results/test_voltage_all_n10000_d7_s2_errors.csv"
current_result_path = project_root / "data/results/test_current_all_n10000_d7_s2_errors.csv"
result_path = project_root / "data/results/test_both_all_n10000_d7_s2_errors.csv"

label_names = np.load(project_root / "data/processed/signal_all_n10000.npz")["label_names"].tolist()

voltage_result = pd.read_csv(voltage_result_path)
current_result = pd.read_csv(current_result_path)
results = voltage_result.merge(
    current_result, on=["file_name", "day", "label"], suffixes=("_voltage", "_current"), validate="one_to_one"
)
if not (len(results) == len(voltage_result) == len(current_result)):
    raise ValueError(f"評価データが一致しない: voltage {len(voltage_result)}, current {len(current_result)}, 共通 {len(results)}")

results["error"] = (
    results["error_voltage"] / results["error_voltage"].median()
    + results["error_current"] / results["error_current"].median()
)
test_labels = results["label"].map(label_names.index).to_numpy()
is_ng = test_labels >= 2
errors = results["error"].to_numpy()
print(f"test {len(results)} (OK {(~is_ng).sum()}, NG {is_ng.sum()})")


def auc(scores, positive):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def print_aucs(scores):
    ok_mask = test_labels <= 1
    print(f"  AUC (全NG)   {auc(scores, is_ng):.3f}")
    for label in (2, 3):
        mask = ok_mask | (test_labels == label)
        print(f"  AUC ({label_names[label]}) {auc(scores[mask], test_labels[mask] == label):.3f}")


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


for channel in ("voltage", "current"):
    print(f"\n[{channel} だけ]")
    print_aucs(results[f"error_{channel}"].to_numpy())

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
results.sort_values("error", ascending=False).to_csv(result_path, index=False)
print(f"saved -> {result_path}")
