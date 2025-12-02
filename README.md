# README

The goal of this project is to create a multi-modal conten moderation system.

We will train 3 different models:
1. Text-only model
2. Image-only model
3. Multimodal model (for stuff like memes)


The output of the models will be unified to follow this schema:

Text-only schema:
```
{
    "id": str,
    "text": str,
    "toxicity": int,  # 0/1
    "hate": int,      # 0/1
    "source": str,    # 'jigsaw'
}
```

Multimodal schema:
```
{
    "id": str,
    "text": str,
    "image_path": str,  # relative path under data/raw/...
    "hate": int,        # 0/1
    "source": str,      # 'hateful_memes' or 'mmhs150k'
}
```

## Datasets

We will start with three datasets (and maybe incorporate more as we build):
1. Jigsaw Toxic Comments (text-only)
2. MMHS150K (multimodal)
3. Facebook Hateful Memes Dataset (multimodal)

