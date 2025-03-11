import logging
import argparse
import pandas as pd
import sys
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score

from format_checker import check_format

def evaluate(pred_file: str, gold_file: str):
    """
    Evaluates predicted labels against the gold standard.
    Both files must be in CSV format with columns 'id' and 'label'.

    Args:
        pred_file (str): Path to the prediction CSV file.
        gold_file (str): Path to the ground-truth CSV file.

    Returns:
        tuple: (macro_f1, micro_f1, accuracy, precision_macro, recall_macro, precision_micro, recall_micro, classification_report)
    """
    pred_df = pd.read_csv(pred_file)[['id', 'label']]
    gold_df = pd.read_csv(gold_file)[['id', 'label']]
    
    merged_df = pd.merge(pred_df, gold_df, on='id', suffixes=('_pred', '_gold'))
    
    macro_f1 = f1_score(merged_df['label_gold'], merged_df['label_pred'], average='macro', zero_division=0)
    micro_f1 = f1_score(merged_df['label_gold'], merged_df['label_pred'], average='micro', zero_division=0)
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    
    precision_macro = precision_score(merged_df['label_gold'], merged_df['label_pred'], average='macro', zero_division=0)
    recall_macro = recall_score(merged_df['label_gold'], merged_df['label_pred'], average='macro', zero_division=0)
    precision_micro = precision_score(merged_df['label_gold'], merged_df['label_pred'], average='micro', zero_division=0)
    recall_micro = recall_score(merged_df['label_gold'], merged_df['label_pred'], average='micro', zero_division=0)
    
    report = classification_report(merged_df['label_gold'], merged_df['label_pred'], zero_division=0, digits=4)
    
    return macro_f1, micro_f1, accuracy, precision_macro, recall_macro, precision_micro, recall_micro, report

def validate_file(file_path: str) -> bool:
    if not check_format(file_path):
        logging.error(f"Bad format for file {file_path}. Cannot evaluate.")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--gold_file_path", "-g", type=str, required=True,
                        help="Path to the CSV file with gold labels (columns: id, label).")
    parser.add_argument("--prediction_file_path", "-p", type=str, required=True,
                        help="Path to the CSV file with predicted labels (columns: id, label).")
    args = parser.parse_args()
    
    if validate_file(args.prediction_file_path) and validate_file(args.gold_file_path):
        logging.info("Prediction and gold file formats are correct.")
        (macro_f1, micro_f1, accuracy,
         precision_macro, recall_macro,
         precision_micro, recall_micro,
         report) = evaluate(args.prediction_file_path, args.gold_file_path)
        logging.info(f"macro-F1 = {macro_f1:.5f}\tmicro-F1 = {micro_f1:.5f}\taccuracy = {accuracy:.5f}")
        logging.info(f"Macro Precision = {precision_macro:.5f}\tMacro Recall = {recall_macro:.5f}")
        logging.info(f"Micro Precision = {precision_micro:.5f}\tMicro Recall = {recall_micro:.5f}")
        logging.info("\nClassification Report:\n" + report)
    else:
        logging.error("One or more files have an incorrect format. Exiting.")
