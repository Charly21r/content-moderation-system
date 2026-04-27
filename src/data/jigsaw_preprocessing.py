import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import get_settings
from src.data.counterfactual_augmentation import augment_with_counterfactuals

logger = logging.getLogger(__name__)

_settings = get_settings()

RAW_PATH = Path(_settings.data.raw_path)
OUT_DIR = Path(_settings.data.preprocessed_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = _settings.training.seed
TEST_SIZE = _settings.data.test_size
VAL_SIZE = _settings.data.val_size


def load_jigsaw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df


def map_to_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    toxicity_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    hate_cols = ["identity_hate"]

    df["toxicity"] = (df[toxicity_cols].sum(axis=1) > 0).astype(int)
    df["hate"] = (df[hate_cols].sum(axis=1) > 0).astype(int)

    df["safe"] = ((df["toxicity"] == 0) & (df["hate"] == 0)).astype(int)

    return df[["comment_text", "toxicity", "hate", "safe"]].rename(columns={"comment_text": "text"})


def stratified_split(df: pd.DataFrame):
    # Create temporal new label for stratification
    df["strat_tmp_column"] = df["toxicity"].astype(str) + df["hate"].astype(str)

    # Split stratifying by the toxicity label
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["strat_tmp_column"], random_state=RANDOM_STATE
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["strat_tmp_column"],
        random_state=RANDOM_STATE,
    )

    logger.info("Train hate rate: %.4f", train_df["hate"].mean())
    logger.info("Val hate rate: %.4f", val_df["hate"].mean())
    logger.info("Test hate rate: %.4f", test_df["hate"].mean())

    # Drop the temporal column
    train_df = train_df.drop(columns=["strat_tmp_column"])
    val_df = val_df.drop(columns=["strat_tmp_column"])
    test_df = test_df.drop(columns=["strat_tmp_column"])

    return train_df, val_df, test_df


def main():
    df = load_jigsaw()
    df_unified = map_to_policy_labels(df)
    df_train, df_val, df_test = stratified_split(df_unified)

    df_train = augment_with_counterfactuals(
        df_train, text_col="text", max_augment_per_sample=1, frac=0.7, random_state=RANDOM_STATE
    )

    df_train.to_csv(OUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUT_DIR / "test.csv", index=False)

    # Save the unified data to perform initial EDA
    df_unified.to_csv(OUT_DIR / "jigsaw_unified.csv")

    logger.info("Saved train, test, and val data to %s", OUT_DIR)


if __name__ == "__main__":
    main()
