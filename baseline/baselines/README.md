# Shared Task Baselines Usage Instructions (All Files in CSV Format)

This document explains how to run the three baselines for the shared task. **All input files are expected to be in CSV format.**

---

## 1. Logistic Regression (LR) Baseline

This baseline uses TF‑IDF features with a Logistic Regression classifier. The training and development CSV files should have the following columns:
- `id`
- `text`
- `label`

The test CSV file should have the following columns:
- `id`
- `text`

**Example command:**

```bash
python lr_baseline.py \
  --train_file_path ../sample/train.csv \
  --dev_file_path ../sample/dev.csv \
  --test_file_path ../sample/test.csv \
  --prediction_file_path ../output/predictions_lr.csv \
  --ngram_range 1,2 \
  --analyzer word \
```

*Notes:*
- If no development file is provided, the script will split the training data using the `--dev_size` ratio.
- You can adjust the n-gram range and analyzer (for example, use `char` for character n-grams).

---

## 2. Transformer Model Baseline

This baseline fine‑tunes a Transformer model using the Hugging Face Transformers library. All files are in CSV format. The training and development CSV files should include:
- `id`
- `text`
- `label`

The test CSV file should include:
- `id`
- `text`

**Example command:**

```bash
python transformer_finetune.py \
  --train_file_path ../sample/train.csv \
  --dev_file_path ../sample/dev.csv \
  --test_file_path ../sample/test.csv \
  --subtask A \
  --model xlm-roberta-base \
  --prediction_file_path ../output/predictions_transformer.csv
  --epochs 4
```

*Notes:*
- If a development file is not supplied, the script will automatically split the training set using a default dev ratio (e.g., 20%).
- This baseline uses only the labels "human" and "machine". Adjust the `--subtask` argument as needed.

---

## 3. Binoculars Zero-Shot Detection Baseline

This baseline implements a zero‑shot detection method (Binoculars) that does not require any training. It compares the outputs of two causal language models (an observer and a performer) to compute a score and classify each input as `"human"` or `"machine"`. **All input files must be in CSV format.**

For the test CSV file, only the following columns are required:
- `id`
- `text`

**Example command:**

```bash
python binoculars_baseline.py \
  --test_file_path ../sample/test.csv \
  --prediction_file_path ../output/predictions_binoculars.csv \
  --observer_model tiiuae/Falcon3-3B-Base \
  --performer_model tiiuae/Falcon3-3B-Instruct \
  --mode low-fpr \
  --max_token 512 \
  --use_bfloat16 \
  --low_fpr_threshold 0.85 \
  --accuracy_threshold 0.90
```

*Notes:*
- The `--mode` option allows you to choose between `"low-fpr"` (optimized for a low false-positive rate) or `"accuracy"` thresholds.
- The observer and performer models can be replaced with other model IDs if needed.
- This method requires that the test file has only the `id` and `text` columns.

---

## Summary

- **LR Baseline:** Uses TF‑IDF and Logistic Regression.
- **Transformer Baseline:** Fine‑tunes a Transformer model using CSV files for training, development, and testing.
- **Binoculars Baseline:** A zero‑shot method that computes a score from two models and classifies inputs as "human" or "machine" based on a threshold.

Adjust file paths and parameters as needed for your shared task setup.