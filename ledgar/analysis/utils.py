import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import csv

def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """return micro/macro-F1 and accuracy"""
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"micro_f1": float(micro), "macro_f1": float(macro), "accuracy": float(acc)}

def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int]=None) -> np.ndarray:
    """return F1 per class，shape = [C]。"""
    if num_classes is None:
        num_classes = int(np.max([y_true.max(), y_pred.max()])) + 1
    f1s = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        f1s[c] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return f1s

def aggregate_over_seeds(
    y_true: np.ndarray,
    preds_base_seeds: List[np.ndarray],
    preds_sra_seeds: List[np.ndarray],
    out_prefix: str = "seed_agg"
) -> Dict[str, Dict[str, float]]:
    def collect(preds_list):
        vals = {"micro_f1": [], "macro_f1": [], "accuracy": []}
        for preds in preds_list:
            m = metrics_from_preds(y_true, preds)
            for k in vals.keys():
                vals[k].append(m[k])
        return {k: np.array(v) for k, v in vals.items()}

    base = collect(preds_base_seeds)
    sra  = collect(preds_sra_seeds)

    summary = {}
    for metric in ["micro_f1", "macro_f1", "accuracy"]:
        summary[f"base_{metric}_mean"] = float(base[metric].mean())
        summary[f"base_{metric}_std"]  = float(base[metric].std(ddof=1)) if len(base[metric])>1 else 0.0
        summary[f"sra_{metric}_mean"]  = float(sra[metric].mean())
        summary[f"sra_{metric}_std"]   = float(sra[metric].std(ddof=1)) if len(sra[metric])>1 else 0.0
        summary[f"delta_{metric}_mean"]= float((sra[metric]-base[metric]).mean())
        summary[f"delta_{metric}_std"] = float((sra[metric]-base[metric]).std(ddof=1)) if len(sra[metric])>1 else 0.0

    x = np.arange(3)
    base_means = [summary["base_micro_f1_mean"], summary["base_macro_f1_mean"], summary["base_accuracy_mean"]]
    sra_means  = [summary["sra_micro_f1_mean"],  summary["sra_macro_f1_mean"],  summary["sra_accuracy_mean"]]
    base_errs  = [summary["base_micro_f1_std"],  summary["base_macro_f1_std"],  summary["base_accuracy_std"]]
    sra_errs   = [summary["sra_micro_f1_std"],   summary["sra_macro_f1_std"],   summary["sra_accuracy_std"]]

    width = 0.35
    plt.figure()
    plt.bar(x - width/2, base_means, width, yerr=base_errs, capsize=4, label="Base")
    plt.bar(x + width/2, sra_means,  width, yerr=sra_errs,  capsize=4, label="SRA")
    plt.xticks(x, ["micro-F1", "macro-F1", "accuracy"])
    plt.ylabel("Score")
    plt.title("Seed-wise mean ± std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_errorbars.png", dpi=200)
    plt.close()

    return summary

# ============ Paired Bootstrap for CI ============

def bootstrap_diff_ci(
    y_true: np.ndarray,
    y_pred_A: np.ndarray,
    y_pred_B: np.ndarray,
    metric: str = "macro_f1",
    B: int = 2000,
    seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    metric choices: "macro_f1" / "micro_f1" / "accuracy"
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def mfun(y, p):
        if metric == "macro_f1":
            return f1_score(y, p, average="macro", zero_division=0)
        elif metric == "micro_f1":
            return f1_score(y, p, average="micro", zero_division=0)
        elif metric == "accuracy":
            return accuracy_score(y, p)
        else:
            raise ValueError("Unknown metric")

    diffs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = mfun(y_true[idx], y_pred_A[idx]) - mfun(y_true[idx], y_pred_B[idx])

    diffs.sort()
    mean = float(diffs.mean())
    lo = float(diffs[int(0.025 * B)])
    hi = float(diffs[int(0.975 * B)])
    return mean, (lo, hi)

# ============ McNemar Test ============
def mcnemar_pvalue(y_true: np.ndarray, y_pred_A: np.ndarray, y_pred_B: np.ndarray, exact: bool = True) -> float:
    A_correct = (y_pred_A == y_true).astype(int)
    B_correct = (y_pred_B == y_true).astype(int)
    n01 = int(np.sum((A_correct == 0) & (B_correct == 1)))
    n10 = int(np.sum((A_correct == 1) & (B_correct == 0)))
    # extreme case
    if n01 + n10 == 0:
        return 1.0
    if exact:
        from math import comb
        n = n01 + n10
        k = min(n01, n10)
        cdf = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p = 2 * cdf
        return min(1.0, p)
    else:
        from math import fabs
        chi2 = (fabs(n01 - n10) - 1)**2 / (n01 + n10)
        import mpmath as mp
        p = 1 - mp.gammainc(0.5, 0, chi2/2) / mp.gamma(0.5)
        return float(p)

# ============ long-tail analysis ============

def make_freq_buckets(
    y_train: np.ndarray,
    num_buckets: int = 3,
    strategy: str = "quantile"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    split into buckets according to frequencies in the training set。
    return (class_to_bucket, bucket_edges)
    """
    C = int(y_train.max()) + 1
    counts = np.bincount(y_train, minlength=C)
    if strategy == "quantile":
        qs = [i/num_buckets for i in range(1, num_buckets)]
        edges = np.quantile(counts, qs)
        class_to_bucket = np.zeros(C, dtype=int)
        for c in range(C):
            class_to_bucket[c] = int(np.sum(counts[c] > edges))
        return class_to_bucket, edges
    else:
        raise ValueError("Only 'quantile' strategy implemented")

def bucket_macro_f1_improvement(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_sra: np.ndarray,
    class_to_bucket: np.ndarray,
    num_buckets: int = 3,
    out_prefix: str = "buckets"
) -> Dict[str, float]:
    C = len(class_to_bucket)
    def macro_f1_on_bucket(bucket_id, preds):
        f1s = []
        for c in range(C):
            if class_to_bucket[c] == bucket_id:
                y_true_bin = (y_true == c).astype(int)
                y_pred_bin = (preds == c).astype(int)
                f1s.append(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        return float(np.mean(f1s)) if f1s else float("nan")

    base_vals, sra_vals, deltas = [], [], []
    for b in range(num_buckets):
        mb = macro_f1_on_bucket(b, y_pred_base)
        mr = macro_f1_on_bucket(b, y_pred_sra)
        base_vals.append(mb); sra_vals.append(mr); deltas.append(mr - mb)

    x = np.arange(num_buckets)
    width = 0.35
    plt.figure()
    plt.bar(x - width/2, base_vals, width, label="Base")
    plt.bar(x + width/2, sra_vals,  width, label="SRA")
    plt.xticks(x, [f"Bucket {i}" for i in range(num_buckets)])
    plt.ylabel("macro-F1")
    plt.title("macro-F1 by frequency bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_macroF1_by_bucket.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(x, deltas)
    plt.xticks(x, [f"Bucket {i}" for i in range(num_buckets)])
    plt.ylabel("Δ macro-F1 (SRA - Base)")
    plt.title("Improvement by bucket")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_improvement_by_bucket.png", dpi=200)
    plt.close()

    return {
        **{f"base_bucket_{i}_macro_f1": base_vals[i] for i in range(num_buckets)},
        **{f"sra_bucket_{i}_macro_f1":  sra_vals[i]  for i in range(num_buckets)},
        **{f"delta_bucket_{i}":         deltas[i]    for i in range(num_buckets)},
    }

def topk_class_improvements(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_sra: np.ndarray,
    class_names: Optional[List[str]] = None,
    k: int = 10,
    out_csv: str = "topk_class_improvements.csv"
) -> List[Tuple[int, float, float, float]]:
    C = int(max(y_true.max(), y_pred_base.max(), y_pred_sra.max())) + 1
    if class_names is None:
        class_names = [str(i) for i in range(C)]
    f1_base = per_class_f1(y_true, y_pred_base, num_classes=C)
    f1_sra  = per_class_f1(y_true, y_pred_sra,  num_classes=C)
    delta   = f1_sra - f1_base
    order   = np.argsort(-delta)

    rows = []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "class_id", "class_name", "base_f1", "sra_f1", "delta"])
        for rank, c in enumerate(order[:k], start=1):
            rows.append((int(c), float(f1_base[c]), float(f1_sra[c]), float(delta[c])))
            w.writerow([rank, c, class_names[c], f1_base[c], f1_sra[c], delta[c]])
    return rows
