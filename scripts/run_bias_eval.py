import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from training.train_text_model import (
    JigsawDataset,
    LABEL_COLS,
    NUM_LABELS,
    MAX_LENGTH,
    compute_metrics
)

ROOT = Path(__file__).resolve().parents[1]
DATA_BIAS_PATH = ROOT / "data" / "bias_eval" / "templated_lexical_bias.csv"
MODEL_DIR = ROOT / "models" / "text_toxicity" / "artifacts"
MODEL_PATH = MODEL_DIR / "model"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
REPORT_PATH = MODEL_DIR / "bias_report.json"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16


def load_model_and_tokenizer(device: torch.device):
    config = AutoConfig.from_pretrained(
        MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME,
        config=config,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME
    )
    return model, tokenizer


def main():
    if not DATA_BIAS_PATH.exists():
        raise FileNotFoundError(f"Bias eval dataset not found at {DATA_BIAS_PATH}")

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Thresholds not found at {THRESHOLDS_PATH}")

    with THRESHOLDS_PATH.open("r") as f:
        thresholds = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(device)

    df_bias = pd.read_csv(DATA_BIAS_PATH)
    bias_ds = JigsawDataset(df_bias, tokenizer, max_length=MAX_LENGTH)
    bias_loader = DataLoader(bias_ds, batch_size=BATCH_SIZE)

    # Simple eval loop
    model.eval()
    all_labels = []
    all_probs = []
    all_texts: list[str] = []

    with torch.no_grad():
        for batch in bias_loader:
            texts = batch["text"]
            labels = batch["labels"].cpu().numpy()

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ["labels", "text"]
            }

            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)
            all_texts.extend(texts)

    all_labels_arr = np.concatenate(all_labels, axis=0)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    report_metrics = compute_metrics(
        all_labels_arr,
        all_probs_arr,
        all_texts,
        thresholds,
    )
    
    with REPORT_PATH.open("w") as f:
        json.dump(report_metrics, f, indent=2)

    print(f"Wrote bias report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
