"""Centralized configuration for the content moderation system.

Loads from config/training.yaml with environment variable overrides.
Env vars use the prefix CMS_ (Content Moderation System), e.g.:
  CMS_TRAINING__EPOCHS=3  (double underscore for nesting)
  CMS_TRAINING__BATCH_SIZE=32
"""

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "training.yaml"


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class ModelConfig(BaseSettings):
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    label_cols: list[str] = ["toxicity", "hate"]
    max_length: int = 256


class TrainingConfig(BaseSettings):
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    seed: int = 42
    mixed_precision: bool = True


class DataConfig(BaseSettings):
    raw_path: Path = Path("data/raw/jigsaw/train.csv")
    preprocessed_dir: Path = Path("data/preprocessed/text")
    test_size: float = 0.2
    val_size: float = 0.15


class PathsConfig(BaseSettings):
    model_dir: Path = Path("models/text_toxicity/artifacts")
    sensitive_words_config: Path = Path("config/local_sensitive_words.json")


class BiasEvalConfig(BaseSettings):
    templated_data_path: Path = Path("data/bias_eval/templated_lexical_bias.csv")
    batch_size: int = 32
    max_fpr_delta: float = 0.05


class Settings(BaseSettings):
    model_config = {"env_prefix": "CMS_", "env_nested_delimiter": "__"}

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    bias_eval: BiasEvalConfig = Field(default_factory=BiasEvalConfig)

    mlflow_tracking_uri: str | None = None


@lru_cache(maxsize=1)
def get_settings(config_path: Path = _DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from YAML config with env var overrides."""
    yaml_data = _load_yaml_config(config_path)
    return Settings(**yaml_data)
