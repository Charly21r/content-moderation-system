# EDA Report  
*Last updated: 2025-12-03*

---

# Jigsaw Dataset EDA
This section summarizes the exploratory data analysis performed on the Jigsaw Toxic Comment dataset after mapping it into a unified moderation schema (safe, toxicity, hate). The goal is to understand dataset characteristics, distributional biases, and modeling implications before training the first model.

## Label Distribution
The dataset shows strong class imbalance, especially for the hate category:

| Label      | Count   | Percentage |
| ---------- | ------- | ---------- |
| Toxicity=1 | 16,171  | ~10%       |
| Hate=1     | 1,405   | <1%        |
| Safe=1     | 143,346 | ~89%       |

**Implications:**
- Both class weighting and stratified sampling will be required. Plus maybe more advanced techniques (SMOTE, etc).
- Accuracy could be misleading, evaluations must use ROC-AUC, F1-score, confusion matrices, etc.

## Text Characteristics
**Text Length:**
- Distributions is heavily skewed, so the majority of the texts are short.
- Majority of examples (99.95%) fit within 256 tokens, so max_length=256 is feasible for BERT without significant truncation.

**Domain Characteristics:**
- The most common n-grams are:
    - article, wikipedia, page, talk

This confirms the dataset comes from Wikipedia talk pages

**Implications:**
- Model may not generalize so well to more informal or meme-style content.
- Additional data would be required for broader coverage.

## Lexical Bias Analysis
Identity-related words were examined for disproportionate association with toxic/hate labels.

| Term   | Safe  | Toxic | Hate  |
| ------ | ----- | ----- | ----- |
| black  | 0.860 | 0.136 | 0.036 |
| white  | 0.870 | 0.127 | 0.036 |
| muslim | 0.874 | 0.120 | 0.060 |
| jew    | 0.818 | 0.171 | 0.089 |
| gay    | 0.431 | 0.558 | 0.287 |
| asian  | 0.880 | 0.114 | 0.052 |

**Observations:**
- Some terms correlate disproportionately with toxic/hate labels.
- "gay" shows a particularly high association with toxic and hateful labels.
- This could reflect a dataset bias rather than actual content patterns.

**Implications:**
- A model trained on this data may over-penalize the mention of these terms even in neutral contexts.
- A fairness mitigation strategy (counterfactual data augmentation, bias regularization) is needed to prevent unintended discrimination.

## Profanity and Slurs Analysis
Profanity and slur keywords were analyzed for their correlation with labels. 

| Keyword            | Safe    | Toxic  | Hate |
| ------------------ | ------- | ------ | ---- |
| *profanity 1*      | 0.001   | 0.264  | 1.00 |
| *profanity 2*      | 0.0004  | 0.053  | 1.00 |
| *profanity 3*      | 0.0003  | 0.041  | 1.00 |
| *slur 1*           | 0.00027 | 0.0136 | 1.00 |
| *slur 2*           | 0.00204 | 0.0024 | 1.00 |
| *slur 3*           | 0.00010 | 0.030  | 1.00 |

**Observations:**
- Slur tokens are perfect predictors of the hate label in this dataset.
- The model may learn shortcut features (e.g., "if slur -> hate").
- It may fail to detect implicit or coded hate without explicit slurs.

**Implications:**
- Additional datasets with subtle/non-explicit hate content are needed.
- Multimodal datasets (Hateful Memes, MMHS150K) will help.


## Label Noise Exploration
318 samples were flagged as potentially noisy, using checks such as:
- hate=1 but toxicity=0
- safe=1 but containing profanity
- Unusually short or long texts
- Duplicates

**Observations:**
- Label noise is present but manageable.
- Some toxicity/hate cases involve sarcasm, quotes, or ambiguous contexts.

**Implications:**
- Consider label smoothing or robust loss functions.
- In future, integrate CleanLab or confident learning approaches.