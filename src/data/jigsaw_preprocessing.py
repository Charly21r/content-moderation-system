from pathlib import Path
import pandas as pd

RAW_PATH = Path('data/raw/jigsaw/train.csv')
OUT_DIR = Path('data/preprocessed/text')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_jigsaw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df

def map_to_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    toxicity_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    hate_cols = ["identity_hate"]

    df["toxicity"] = (df[toxicity_cols].sum(axis=1) > 0).astype(int)
    df["hate"] = (df[hate_cols].sum(axis=1) > 0).astype(int)
    
    df["safe"] = ( (df["toxicity"] == 0) & (df["hate"] == 0) ).astype(int)

    return df[["comment_text", "toxicity", "hate", "safe"]].rename(
        columns={"comment_text": "text"}
    )

def main():
    df = load_jigsaw()
    df_unified = map_to_policy_labels(df)
    out_path = OUT_DIR / "jigsaw_unified.csv"
    df_unified.to_csv(out_path, index=False)
    print(f"Saved unified jigsaw data to {out_path}")


if __name__ == "__main__":
    main()