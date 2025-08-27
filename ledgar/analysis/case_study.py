import re
import numpy as np
import pandas as pd
from os.path import abspath
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

BASELINE_CKPT = ""
SRA_CKPT      = ""
SRA_TEST_DIR  = ""

MAX_LEN = 512
BATCH   = 64

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def extract_retrieved(aug_text: str) -> str:
    if "Related clause for reference:" in aug_text:
        part = aug_text.split("Related clause for reference:", 1)[1]
        return part.strip()
    if "Related clause:" in aug_text:
        part = aug_text.split("Related clause:", 1)[1]
        return part.strip()
    m = re.search(r"Related clause[^:]*:\s*(.*)$", aug_text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def build_trainer(model_dir, tok):
    args = TrainingArguments(
        output_dir=f"{abspath(model_dir)}/_pred_tmp",
        per_device_eval_batch_size=BATCH,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        fp16=True,
        fp16_full_eval=True,
    )
    data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    model = AutoModelForSequenceClassification.from_pretrained(abspath(model_dir), local_files_only=True)
    return Trainer(model=model, args=args, tokenizer=tok, data_collator=data_collator)

def encode_column(ds, tok, col):
    return ds.map(lambda b: tok(b[col], truncation=True, padding="max_length", max_length=MAX_LEN),
                  batched=True, remove_columns=[c for c in ds.column_names if c != "label"])

def main():
    raw_test = load_dataset("lex_glue", "ledgar", split="test")   # columns: text, label
    sra_test = load_from_disk(abspath(SRA_TEST_DIR))              # columns: augmented_text, label

    assert len(raw_test) == len(sra_test)
    y_true_raw = np.array(raw_test["label"], dtype=int)
    y_true_sra = np.array(sra_test["label"], dtype=int)
    assert np.array_equal(y_true_raw, y_true_sra)

    # --- baseline prediciton ---
    tok_base = AutoTokenizer.from_pretrained(abspath(BASELINE_CKPT), use_fast=True, local_files_only=True)
    tr_base  = build_trainer(BASELINE_CKPT, tok_base)
    enc_base = encode_column(raw_test, tok_base, col="text")
    logits_base = tr_base.predict(enc_base).predictions
    probs_base  = softmax(logits_base)
    pred_base   = probs_base.argmax(axis=1)

    # --- SRA prediction ---
    tok_sra = AutoTokenizer.from_pretrained(abspath(SRA_CKPT), use_fast=True, local_files_only=True)
    tr_sra  = build_trainer(SRA_CKPT, tok_sra)
    assert "augmented_text" in sra_test.column_names
    enc_sra = encode_column(sra_test, tok_sra, col="augmented_text")
    logits_sra = tr_sra.predict(enc_sra).predictions
    probs_sra  = softmax(logits_sra)
    pred_sra  = probs_sra.argmax(axis=1)

    gold = y_true_raw
    base_correct = (pred_base == gold)
    sra_correct  = (pred_sra  == gold)

    success_idx = np.where((~base_correct) & (sra_correct))[0]  # baseline wrong & sra correct
    failure_idx = np.where((base_correct) & (~sra_correct))[0]  # baseline wrong & sra correct

    margin_success = probs_sra[np.arange(len(gold)), gold] - probs_base[np.arange(len(gold)), gold]
    margin_failure = probs_base[np.arange(len(gold)), gold] - probs_sra[np.arange(len(gold)), gold]

    def pick_cases(idxs, margin, topk=10):
        idxs = np.asarray(idxs)
        if idxs.size == 0:
            return []
        ords = np.argsort(margin[idxs])[::-1][:topk]
        return [int(x) for x in idxs[ords]]


    pick_succ = pick_cases(success_idx, margin_success, topk=10)
    pick_fail = pick_cases(failure_idx, margin_failure, topk=10)

    def rows_from_indices(idxs):
        rows = []
        idxs = [int(x) for x in idxs]
        aug_col_exists = "augmented_text" in sra_test.column_names

        def recover_text_from_aug(s):
            if "Original clause:" in s:
                t = s.split("Related clause", 1)[0]
                t = t.replace("Original clause:", "").replace("</s>", "").strip()
                return t
            return ""

        for i in idxs:
            text = raw_test[i]["text"]
            aug  = sra_test[i]["augmented_text"] if aug_col_exists else ""
            retrieved = extract_retrieved(aug) if aug else ""
            rows.append({
                "idx": i,
                "gold": int(gold[i]),
                "baseline_pred": int(pred_base[i]),
                "sra_pred": int(pred_sra[i]),
                "baseline_prob_gold": float(probs_base[i, gold[i]]),
                "sra_prob_gold": float(probs_sra[i, gold[i]]),
                "text": text if text else recover_text_from_aug(aug),
                "augmented_text": aug,
                "retrieved_clause": retrieved[:500],
            })
        return rows

    succ_rows = rows_from_indices(pick_succ)
    fail_rows = rows_from_indices(pick_fail)

    pd.DataFrame(succ_rows).to_csv("cases_success.csv", index=False)
    pd.DataFrame(fail_rows).to_csv("cases_failure.csv", index=False)
    print(f"Saved cases_success.csv ({len(succ_rows)})  and  cases_failure.csv ({len(fail_rows)})")

if __name__ == "__main__":
    main()
