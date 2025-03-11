# Evaluation Scripts Usage Instructions

This document provides instructions on how to check the format of your CSV prediction files and evaluate your predictions against the gold standard using the provided scripts.

## Files

- **format_checker.py**:  
  Checks that a CSV file is in the correct format. The file must contain two columns: `id` and `label`. For subtask A, valid labels are 0 or 1.

- **scorer.py**:  
  Evaluates predicted labels against the ground-truth labels. Both files must be in CSV format with columns `id` and `label`.

## Usage

### 1. Format Checker

To verify that your prediction file is in the correct format, run:

```bash
python format_checker.py --prediction_file_path path/to/predictions.csv
```

*Example:*

```bash
python format_checker.py --prediction_file_path predictions.csv
```

### 2. Scorer

To evaluate your predictions against the ground-truth labels, run:

```bash
python scorer.py --gold_file_path path/to/groundtruth.csv --prediction_file_path path/to/predictions.csv
```

*Example:*

```bash
python scorer.py --gold_file_path groundtruth.csv --prediction_file_path predictions.csv
```

The scorer will output the following metrics:
- Macro F1 Score
- Micro F1 Score
- Accuracy

## Summary

- Use **format_checker.py** to ensure your CSV files contain the required `id` and `label` columns with valid labels (0 or 1).
- Use **scorer.py** to compute evaluation metrics for your predictions against the gold standard.
