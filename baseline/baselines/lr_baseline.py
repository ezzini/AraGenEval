import pandas as pd
import numpy as np
import argparse
import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

def get_data(train_path, dev_path, test_path, random_seed, dev_size=0.2):
    """
    Read CSV files for training, development, and (optionally) testing.
    The train and dev files are expected to have columns: id, text, label.
    If a dev file is provided, it is used; otherwise, the training set is split
    using the dev_size ratio.
    The test file is assumed to have only the columns: id and text.
    """
    train_df = pd.read_csv(train_path)
    
    if dev_path is not None:
        dev_df = pd.read_csv(dev_path)
    else:
        train_df, dev_df = train_test_split(
            train_df, test_size=dev_size, stratify=train_df['label'], random_state=random_seed
        )
    
    test_df = pd.read_csv(test_path) if test_path is not None else None
    return train_df, dev_df, test_df

def train_ml_model(train_df, dev_df, ngram_range=(1, 2), analyzer="word"):
    """
    Fit a TF-IDF vectorizer on the training texts using the specified ngram_range
    and analyzer, then train a Logistic Regression classifier.
    The dev set is used for evaluation (with multiple metrics printed).
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    
    # Fit the vectorizer on training texts and transform train/dev texts
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    X_dev = vectorizer.transform(dev_df['text'])
    y_dev = dev_df['label']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate on dev set
    y_dev_pred = clf.predict(X_dev)
    
    acc = accuracy_score(y_dev, y_dev_pred)
    precision = precision_score(y_dev, y_dev_pred, average='micro')
    recall = recall_score(y_dev, y_dev_pred, average='micro')
    f1_micro = f1_score(y_dev, y_dev_pred, average='micro')
    f1_macro = f1_score(y_dev, y_dev_pred, average='macro')
    
    print("Dev Accuracy:", acc)
    print("Dev Precision (micro):", precision)
    print("Dev Recall (micro):", recall)
    print("Dev F1 Score (micro):", f1_micro)
    print("Dev F1 Score (macro):", f1_macro)
    print("\nClassification Report on Dev Set:")
    print(classification_report(y_dev, y_dev_pred, digits=4))
    
    return vectorizer, clf

def test_model(test_df, vectorizer, clf):
    """
    Transform the test data using the fitted TF-IDF vectorizer and predict labels.
    Since the test file contains only id and text, no probabilities or metrics are computed.
    """
    X_test = vectorizer.transform(test_df['text'])
    preds = clf.predict(X_test)
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True,
                        help="Path to the train file (CSV with columns: id, text, label).", type=str)
    parser.add_argument("--dev_file_path", "-d", default=None,
                        help="Optional path to the dev file (CSV with columns: id, text, label).", type=str)
    parser.add_argument("--test_file_path", "-t", default=None,
                        help="Optional path to the test file (CSV with columns: id, text).", type=str)
    parser.add_argument("--dev_size", "-ds", default=0.2, type=float,
                        help="If no dev file is provided, use this fraction of train for dev.")
    parser.add_argument("--ngram_range", "-ng", default="1,2", type=str,
                        help="Ngram range as two comma-separated integers (e.g., '1,2' for unigrams and bigrams).")
    parser.add_argument("--analyzer", "-a", default="word", choices=["word", "char"], type=str,
                        help="Analyzer type for TF-IDF: 'word' for word ngrams, 'char' for character ngrams.")
    parser.add_argument("--prediction_file_path", "-p", default=None, type=str,
                        help="Path where to save the prediction file (CSV format).")
    
    args = parser.parse_args()
    
    # Use args directly without assigning to extra variables.
    # Check for file existence.
    if not os.path.exists(args.train_file_path):
        logging.error("Train file doesn't exist: {}".format(args.train_file_path))
        raise ValueError("Train file doesn't exist: {}".format(args.train_file_path))
    if args.dev_file_path is not None and not os.path.exists(args.dev_file_path):
        logging.error("Dev file doesn't exist: {}".format(args.dev_file_path))
        raise ValueError("Dev file doesn't exist: {}".format(args.dev_file_path))
    if args.test_file_path is not None and not os.path.exists(args.test_file_path):
        logging.error("Test file doesn't exist: {}".format(args.test_file_path))
        raise ValueError("Test file doesn't exist: {}".format(args.test_file_path))
    
    # Load data and create train/dev/test splits using args directly.
    train_df, dev_df, test_df = get_data(args.train_file_path, args.dev_file_path, args.test_file_path, random_seed=0, dev_size=args.dev_size)
    
    # Define label mapping and convert labels directly.
    label2id = {"human": 0, "machine": 1}
    id2label = {0: "human", 1: "machine"}
    train_df["label"] = train_df["label"].map(label2id)
    dev_df["label"] = dev_df["label"].map(label2id)
    if test_df is not None and "label" in test_df.columns:
        test_df["label"] = test_df["label"].map(label2id)
    
    # Parse ngram_range directly from args.
    ngram_range = tuple(map(int, args.ngram_range.split(',')))
    
    # Train the ML model using TF-IDF features and Logistic Regression.
    vectorizer, clf = train_ml_model(train_df, dev_df, ngram_range=ngram_range, analyzer=args.analyzer)
    
    # If a test set is provided, run the test phase and dump predictions.
    if test_df is not None:
        preds = test_model(test_df, vectorizer, clf)
        pred_labels = [id2label[pred] for pred in preds]
    
        if args.prediction_file_path is not None:
            predictions_df = pd.DataFrame({
                'id': test_df['id'],
                'label': pred_labels
            })
            predictions_df.to_csv(args.prediction_file_path, index=False)
            print("Predictions saved to", args.prediction_file_path)
        else:
            print("Test predictions:")
            print(pred_labels)
    else:
        print("No test file provided. Training and evaluation on the dev set is complete.")
