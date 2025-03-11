import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Union, List, Tuple
from utils import assert_tokenizer_consistency, entropy, perplexity  

logging.basicConfig(level=logging.INFO)

DEFAULT_BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
DEFAULT_BINOCULARS_FPR_THRESHOLD = 0.8536432310785527

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class Binoculars:
    """
    Zero-shot detection baseline.
    
    Given an input text, this method computes a score (perplexity/entropy)
    and classifies the input as "machine" if the score is below a threshold,
    and "human" otherwise.
    """
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        low_fpr_threshold: float = DEFAULT_BINOCULARS_FPR_THRESHOLD,
        accuracy_threshold: float = DEFAULT_BINOCULARS_ACCURACY_THRESHOLD,
    ) -> None:
        self._assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        self.change_mode(mode, low_fpr_threshold, accuracy_threshold)
        # Use BitsAndBytesConfig for 8-bit quantization.
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            quantization_config=quant_config,
            device_map=DEVICE_1,
            trust_remote_code=True,
            torch_dtype="auto",
            token=os.environ.get("HF_TOKEN", None)
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            quantization_config=quant_config,
            device_map=DEVICE_2,
            trust_remote_code=True,
            torch_dtype="auto",
            token=os.environ.get("HF_TOKEN", None)
        )
        print("models loaded")
        self.observer_model.eval()
        self.performer_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _assert_tokenizer_consistency(self, observer_path: str, performer_path: str) -> None:
        assert_tokenizer_consistency(observer_path, performer_path)

    def change_mode(self, mode: str, low_fpr_threshold: float, accuracy_threshold: float) -> None:
        if mode == "low-fpr":
            self.threshold = low_fpr_threshold
        elif mode == "accuracy":
            self.threshold = accuracy_threshold
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, texts: List[str]) -> transformers.BatchEncoding:
        batch_size = len(texts)
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        )
        return encodings.to(DEVICE_1)

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[List[str], str]) -> Union[float, List[float]]:
        texts = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(texts)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        ent = entropy(observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id)
        scores = ppl / ent
        return scores[0] if isinstance(input_text, str) else scores.tolist()

    def predict(self, input_text: Union[List[str], str]) -> Union[str, List[str]]:
        # Clear unused GPU memory before prediction.
        torch.cuda.empty_cache()
        scores = np.array(self.compute_score(input_text))
        predictions = np.where(scores < self.threshold, "machine", "human")
        return predictions.tolist() if isinstance(input_text, list) else predictions.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binoculars Zero-Shot Detection Baseline"
    )
    parser.add_argument("--test_file_path", "-t", required=True,
                        help="Path to the test CSV file (with columns: id, text).", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True,
                        help="Path where to save the prediction CSV (with columns: id, label).", type=str)
    parser.add_argument("--observer_model", "-ob", default="tiiuae/falcon-7b",
                        help="Path or model name for the observer model.", type=str)
    parser.add_argument("--performer_model", "-pf", default="tiiuae/falcon-7b-instruct",
                        help="Path or model name for the performer model.", type=str)
    parser.add_argument("--mode", "-m", default="low-fpr", choices=["low-fpr", "accuracy"],
                        help="Detection mode (low-fpr or accuracy).", type=str)
    parser.add_argument("--max_token", "-mt", default=512, type=int,
                        help="Maximum number of tokens to observe.",)
    parser.add_argument("--use_bfloat16", "-bf", action="store_true",
                        help="Use bfloat16 precision if available.")
    parser.add_argument("--low_fpr_threshold", default=DEFAULT_BINOCULARS_FPR_THRESHOLD, type=float,
                        help="Threshold for low-fpr mode.")
    parser.add_argument("--accuracy_threshold", default=DEFAULT_BINOCULARS_ACCURACY_THRESHOLD, type=float,
                        help="Threshold for accuracy mode.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_file_path):
        logging.error(f"Test file does not exist: {args.test_file_path}")
        raise ValueError(f"Test file does not exist: {args.test_file_path}")
    
    test_df = pd.read_csv(args.test_file_path)
    if "id" not in test_df.columns or "text" not in test_df.columns:
        logging.error("Test file must contain 'id' and 'text' columns.")
        raise ValueError("Test file must contain 'id' and 'text' columns.")
    
    binoculars = Binoculars(
        observer_name_or_path=args.observer_model,
        performer_name_or_path=args.performer_model,
        use_bfloat16=args.use_bfloat16,
        max_token_observed=args.max_token,
        mode=args.mode,
        low_fpr_threshold=args.low_fpr_threshold,
        accuracy_threshold=args.accuracy_threshold
    )
    
    test_texts = test_df["text"].tolist()
    preds = binoculars.predict(test_texts)
    
    predictions_df = pd.DataFrame({
        "id": test_df["id"],
        "label": preds
    })
    predictions_df.to_csv(args.prediction_file_path, index=False)
    print(f"Predictions saved to {args.prediction_file_path}")
