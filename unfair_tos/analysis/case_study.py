import re
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import trange

CHECKPOINT   = ""
AUG_TEST_DIR = ""
OUT_CSV      = ""

BATCH        = 64
MAX_LEN      = 512
THRESH       = 0.5
MODE         = "exact"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_retrieved(aug_text: str) -> str:
    if not isinstance(aug_text, str):
        return ""
    if "Related clause for reference:" in aug_text:
        return aug_text.split("Related clause for reference:", 1)[1].strip()
    if "Related clause:" in aug_text:
        return aug_text.split("Related clause:", 1)[1].strip()
    m = re.search(r"Related clause[^:]*:\s*(.*)$", aug_text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def batched_probs(texts, tok, model, batch=BATCH, max_len=MAX_LEN):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in trange(0, len(texts), batch, desc="Infer"):
            enc = tok(
                texts[i:i+batch],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(DEVICE)
            logits = model(**enc).logits
            p = torch.sigmoid(logits).cpu().numpy()  # UNFAIR-ToS: 多标签
            probs.append(p)
    return np.vstack(probs)

def to_set(prob_row, thr=THRESH):
    return set(np.where(prob_row >= thr)[0].tolist())

def correct_mask(pred_sets, gold_sets, mode="exact"):
    if mode == "exact":
        return np.array([p == g for p, g in zip(pred_sets, gold_sets)], dtype=bool)
    elif mode == "subset":
        return np.array([g.issubset(p) for p, g in zip(pred_sets, gold_sets)], dtype=bool)
    else:
        raise ValueError("MODE must be 'exact' or 'subset'")

def main():
    raw = load_dataset("lex_glue", "unfair_tos", split="test")
    aug = load_from_disk(AUG_TEST_DIR)

    assert len(raw) == len(aug)
    assert all(set(a) == set(b) for a, b in zip(raw["labels"], aug["labels"]))

    gold_sets = [set(g) for g in raw["labels"]]

    tok = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, local_files_only=True).to(DEVICE)
    model.config.problem_type = "multi_label_classification"

    probs_base = batched_probs(raw["text"], tok, model)
    probs_sra  = batched_probs(aug["augmented_text"], tok, model)

    pred_base = [to_set(row, THRESH) for row in probs_base]
    pred_sra  = [to_set(row, THRESH) for row in probs_sra]

    base_ok = correct_mask(pred_base, gold_sets, mode=MODE)
    sra_ok  = correct_mask(pred_sra,  gold_sets, mode=MODE)
    success_idx = np.where((~base_ok) & (sra_ok))[0]
    print(f"[{MODE}] number of samples with baseline wrong and SRA correct: {len(success_idx)}")

    def gold_prob(prob_row, gset):
        return (1.0 - prob_row.mean()) if len(gset) == 0 else float(prob_row[list(gset)].mean())
    base_gold = np.array([gold_prob(probs_base[i], gold_sets[i]) for i in range(len(raw))])
    sra_gold  = np.array([gold_prob(probs_sra[i],  gold_sets[i]) for i in range(len(raw))])
    gain      = sra_gold - base_gold
    order     = success_idx[np.argsort(gain[success_idx])[::-1]]

    raw_texts = list(raw["text"])
    aug_texts = list(aug["augmented_text"])

    rows = []
    for i in order:
        ii = int(i)
        rows.append({
            "idx": ii,
            "gold": ",".join(map(str, sorted(gold_sets[ii]))),
            "baseline_pred": ",".join(map(str, sorted(pred_base[ii]))),
            "sra_pred": ",".join(map(str, sorted(pred_sra[ii]))),
            "baseline_prob_gold": float(base_gold[ii]),
            "sra_prob_gold": float(sra_gold[ii]),
            "prob_gain": float(gain[ii]),
            "text": raw_texts[ii],
            "augmented_text": aug_texts[ii],
            "retrieved_clause": extract_retrieved(aug_texts[ii])[:600],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(rows)} examples to {OUT_CSV}")


if __name__ == "__main__":
    main()
