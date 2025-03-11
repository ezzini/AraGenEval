import pandas as pd
import numpy as np
import argparse
import logging
import os
import json
import datetime
import sklearn.metrics as skm

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
    set_seed
)
from scipy.special import softmax
from sklearn.model_selection import train_test_split

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)

def get_data(train_path, dev_path, test_path, random_seed, dev_size=0.2):
    """
    Read data files for training, development, and (optionally) testing.
    Files are assumed to be in CSV format with columns: id, text, label.
    If a dev file is provided, it is used; otherwise, the training set is split using dev_size.
    If the test file is not provided, test_df is set to None.
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = skm.accuracy_score(labels, preds)
    precision = skm.precision_score(labels, preds, average='micro')
    recall = skm.recall_score(labels, preds, average='micro')
    f1_micro = skm.f1_score(labels, preds, average='micro')
    f1_macro = skm.f1_score(labels, preds, average='macro')
    #report = skm.classification_report(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        #"classification_report": report
    }

def fine_tune(train_df, valid_df, args):

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
       args.model, num_labels=len(args.label2id), id2label=args.id2label, label2id=args.label2id
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    model.resize_token_embeddings(len(tokenizer))
    
    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.checkpoints_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=args.fp16,
        push_to_hub=False,
        push_to_hub_model_id=None,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    best_model_path = os.path.join(args.checkpoints_path, 'best')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    trainer.save_model(best_model_path)

def test(test_df, model_path, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    
    test_dataset = Dataset.from_pandas(test_df[['text']])
    tokenized_test_dataset = test_dataset.map(
        preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # For test, we don't compute metrics because the file contains no labels.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )
    
    predictions = trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", required=True, type=str,
                        help="Path to the train file (CSV with columns: id, text, label).")
    parser.add_argument("--dev_file_path", default=None, type=str,
                        help="Optional path to the dev file (CSV with columns: id, text, label).")
    parser.add_argument("--test_file_path", default=None, type=str,
                        help="Optional path to the test file (CSV with columns: id, text).")
    parser.add_argument("--model", required=True, type=str,
                        help="Transformer model to train and test.")
    parser.add_argument("--dev_size", default=0.2, type=float,
                        help="Fraction of train to use as dev if a dev file is not provided.")
    parser.add_argument("--prediction_file_path", default=None, type=str,
                        help="Path where to save the prediction file (CSV format).")
    parser.add_argument("--subtask", required=True, type=str, choices=['A', 'B'])
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="Learning rate for training.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 (mixed precision) training.")
    
    args = parser.parse_args()
    
    # Set label mapping
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    args.id2label = id2label
    args.label2id = label2id
    args.checkpoints_path = f"../output/models/{args.model}/subtask{args.subtask}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    train_df, valid_df, test_df = get_data(args.train_file_path, args.dev_file_path, args.test_file_path, random_seed=0, dev_size=args.dev_size)
    
    # Map text labels for train and dev
    train_df["label"] = train_df["label"].map(label2id)
    valid_df["label"] = valid_df["label"].map(label2id)
    # Test file is expected to contain only id and text; no mapping required.
    
    # Train the model using provided args
    fine_tune(train_df, valid_df, args)
    
    # If test data is provided, run predictions and dump to CSV
    if test_df is not None:
        preds = test(test_df, os.path.join(args.checkpoints_path, 'best'), id2label, label2id)
        pred_labels = [id2label[pred] for pred in preds]
        predictions_df = pd.DataFrame({
            'id': test_df['id'],
            'label': pred_labels
        })
        if args.prediction_file_path:
            predictions_df.to_csv(args.prediction_file_path, index=False)
            print("Predictions saved to", args.prediction_file_path)
        else:
            print("Test predictions:")
            print(predictions_df)
    else:
        print("No test file provided. Training and evaluation on the dev set is complete.")
