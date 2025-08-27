import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from utils import (
    aggregate_over_seeds, bootstrap_diff_ci, mcnemar_pvalue,
    make_freq_buckets, bucket_macro_f1_improvement, topk_class_improvements
)
from os.path import abspath

def predict_from_ckpt(model_dir, save_path, sra_dataset_dir=None, max_length=512, eval_bs=64):
    model_dir = abspath(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    tok   = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)

    if sra_dataset_dir is None:
        test_ds = load_dataset("lex_glue", "ledgar", split="test")
        y_true = np.array(test_ds["label"], dtype=int)
        np.save("y_true.npy", y_true)
        print("Saved y_true.npy", y_true.shape)

        def preprocess(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=max_length)

        test_enc = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

    else:
        test_ds = load_from_disk(abspath(sra_dataset_dir))
        assert "augmented_text" in test_ds.column_names
        assert "label" in test_ds.column_names

        y_true = np.array(test_ds["label"], dtype=int)
        np.save("y_true.npy", y_true)
        print("Saved y_true.npy", y_true.shape)

        def preprocess(batch):
            return tok(batch["augmented_text"], truncation=True, padding="max_length", max_length=max_length)

        test_enc = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

    args = TrainingArguments(
        output_dir=f"{model_dir}/_pred_tmp",
        per_device_eval_batch_size=eval_bs,
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

    trainer = Trainer(model=model, args=args, tokenizer=tok, data_collator=data_collator)
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks if cb.__class__.__name__ != "WandbCallback"
    ]

    logits = trainer.predict(test_enc).predictions
    preds = logits.argmax(axis=1)
    np.save(save_path, preds)
    print(f"Saved {save_path}, shape={preds.shape}")

if __name__ == "__main__":
    BASELINE_MODEL_DIRS = [
        "",
    ]
    SRA_MODEL_DIRS = [
        "",
    ]

    SRA_TEST_DIR = ""

    assert len(BASELINE_MODEL_DIRS) == len(SRA_MODEL_DIRS)

    preds_base_paths, preds_sra_paths = [], []
    for i, (b_dir, r_dir) in enumerate(zip(BASELINE_MODEL_DIRS, SRA_MODEL_DIRS), start=1):
        base_out = f"preds_base_seed{i}.npy"
        sra_out  = f"preds_sra_seed{i}.npy"

        predict_from_ckpt(
            model_dir=b_dir,
            save_path=base_out,
            sra_dataset_dir=None,
            max_length=512,
            eval_bs=64,
        )

        predict_from_ckpt(
            model_dir=r_dir,
            save_path=sra_out,
            sra_dataset_dir=SRA_TEST_DIR,
            max_length=512,
            eval_bs=64,
        )

        preds_base_paths.append(base_out)
        preds_sra_paths.append(sra_out)

    y_true = np.load("y_true.npy")

    preds_base_seeds = [np.load(p) for p in preds_base_paths]
    preds_sra_seeds  = [np.load(p) for p in preds_sra_paths]

    summary = aggregate_over_seeds(y_true, preds_base_seeds, preds_sra_seeds, out_prefix="ledgar")
    print("Summary:", summary)

    # mean_diff_micro, (lo_micro, hi_micro) = bootstrap_diff_ci(y_true, preds_sra_seeds[0], preds_base_seeds[0],
    #                                                          metric="micro_f1", B=2000, seed=123)
    # print(f"Δ micro-F1 = {mean_diff_micro:.4f}, 95% CI=({lo_micro:.4f}, {hi_micro:.4f})")

    mean_diff, (lo, hi) = bootstrap_diff_ci(y_true, preds_sra_seeds[0], preds_base_seeds[0],
                                            metric="macro_f1", B=2000, seed=123)
    print(f"Δ macro-F1 = {mean_diff:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")

    from utils import mcnemar_pvalue
    p = mcnemar_pvalue(y_true, preds_sra_seeds[0], preds_base_seeds[0], exact=True)
    print("McNemar p-value:", p)

    train_ds = load_dataset("lex_glue", "ledgar", split="train")
    y_train = np.array(train_ds["label"], dtype=int)

    class_to_bucket, edges = make_freq_buckets(y_train, num_buckets=3, strategy="quantile")
    print("Bucket edges (train freq quantiles):", edges)

    bucket_stats = bucket_macro_f1_improvement(
        y_true=y_true,
        y_pred_base=preds_base_seeds[0],
        y_pred_sra=preds_sra_seeds[0],
        class_to_bucket=class_to_bucket,
        num_buckets=3,
        out_prefix="ledgar_buckets"
    )
    print("Bucket stats:", bucket_stats)

    topk = topk_class_improvements(
        y_true=y_true,
        y_pred_base=preds_base_seeds[0],
        y_pred_sra=preds_sra_seeds[0],
        class_names=None,
        k=15,
        out_csv="ledgar_topk_class_improvements.csv"
    )
    print("Top-k improvements (前5):", topk[:5])
