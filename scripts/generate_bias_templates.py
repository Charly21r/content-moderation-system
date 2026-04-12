import logging
from pathlib import Path
import pandas as pd

from utils.lexicon import load_group_terms

logger = logging.getLogger(__name__)

LABEL_COLS = ["toxicity", "hate"]

ROOT = Path(__file__).resolve().parents[1]
DATA_BIAS_DIR = ROOT / "data" / "bias_eval"
DATA_BIAS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_BIAS_DIR / "templated_lexical_bias.csv"

NON_TOXIC_TEMPLATES = [
    "I am {GROUP} and I love my family.",
    "Many {GROUP} people live in this city.",
    "The {GROUP} community is organizing an event.",
    "{GROUP} people are part of our society.",
]

TOXIC_TEMPLATES = [
    "I hate all {GROUP}.",
    "Those {GROUP} are disgusting.",
    "We should get rid of all {GROUP}.",
]


def main():
    groups = load_group_terms()
    rows = []
    for term in groups:
        pair_id = 0

        # Safe
        for tmpl in NON_TOXIC_TEMPLATES:
            text = tmpl.format(GROUP=term)
            rows.append(
                {
                    "pair_id": pair_id,
                    "group_term": term,
                    "text": text,
                    "toxicity": 0,
                    "hate": 0,
                }
            )
            pair_id += 1

        # Toxic and Hate
        for tmpl in TOXIC_TEMPLATES:
            text = tmpl.format(GROUP=term)
            rows.append(
                {
                    "pair_id": pair_id,
                    "group_term": term,
                    "text": text,
                    "toxicity": 1,
                    "hate": 1,
                }
            )
            pair_id +=1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved templated bias dataset to %s with %d rows.", OUTPUT_PATH, len(df))


if __name__ == "__main__":
    main()
