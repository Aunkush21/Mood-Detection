"""Fine-tune DistilBERT for 4-mood classification on GoEmotions.

Maps the 27 GoEmotions to happy / sad / angry / neutral (same mapping as the
linear baseline in train.py), then fine-tunes distilbert-base-uncased.

Saves the model + tokenizer to ../models/transformer and writes a label map.
Run:  python src/train_transformer.py
"""

import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from train import map_to_mood  # reuse the exact mapping logic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "models", "transformer")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

LABELS = ["angry", "happy", "neutral", "sad"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def to_records(split, label_names):
    return [
        {"text": t, "label": LABEL2ID[map_to_mood(lbl, label_names)]}
        for t, lbl in zip(split["text"], split["labels"])
    ]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    raw = load_dataset("go_emotions")
    names = raw["train"].features["labels"].feature.names

    from datasets import Dataset

    train_ds = Dataset.from_list(
        to_records(raw["train"], names) + to_records(raw["validation"], names)
    )
    test_ds = Dataset.from_list(to_records(raw["test"], names))
    print("Train:", len(train_ds), "Test:", len(test_ds))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "models", "_hf_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\nEval metrics:", metrics)

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    labels = test_ds["label"]
    print("\nFinal accuracy:", accuracy_score(labels, preds))
    print("Macro-F1:", f1_score(labels, preds, average="macro"))
    print("\nClassification report:\n")
    print(classification_report(labels, preds, target_names=LABELS))

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as fh:
        json.dump({"labels": LABELS, "id2label": ID2LABEL}, fh, indent=2)
    print("\nSaved transformer model to", OUT_DIR)


if __name__ == "__main__":
    main()
