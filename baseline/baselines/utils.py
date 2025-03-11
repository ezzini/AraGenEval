import pandas as pd
import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss, Softmax
from transformers import AutoTokenizer
from typing import Union


# Define loss and softmax functions
ce_loss_fn = CrossEntropyLoss(reduction="none")
softmax_fn = Softmax(dim=-1)

def assert_tokenizer_consistency(model_id_1: str, model_id_2: str) -> None:
    """
    Assert that the tokenizers for the given models have identical vocabularies.
    
    Args:
        model_id_1 (str): The identifier or path for the first model.
        model_id_2 (str): The identifier or path for the second model.
    
    Raises:
        ValueError: If the vocabularies of the two tokenizers are not identical.
    """
    tokenizer1 = AutoTokenizer.from_pretrained(model_id_1)
    tokenizer2 = AutoTokenizer.from_pretrained(model_id_2)
    
    if tokenizer1.vocab != tokenizer2.vocab:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute the per-sequence perplexity (cross-entropy loss) based on model logits.

    Args:
        encoding: A BatchEncoding containing 'input_ids' and 'attention_mask'.
        logits: Model logits of shape [batch_size, seq_len, vocab_size].
        median: If True, returns the median loss per sequence; otherwise, returns the mean loss.
        temperature: Temperature scaling factor applied to logits.

    Returns:
        A NumPy array of loss values (one per sequence).
    """
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ce_losses = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
    ce_losses = ce_losses.masked_fill(~shifted_attention_mask.bool(), float("nan"))

    if median:
        loss_values = np.nanmedian(ce_losses.cpu().float().numpy(), axis=1)
    else:
        loss_values = (
            (ce_losses * shifted_attention_mask).sum(1)
            / shifted_attention_mask.sum(1)
        ).to("cpu").float().numpy()

    return loss_values

def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute an aggregated cross-entropy between two distributions derived from p_logits and q_logits.

    Args:
        p_logits: Observer model logits of shape [batch_size, seq_len, vocab_size].
        q_logits: Performer model logits of shape [batch_size, seq_len, vocab_size].
        encoding: BatchEncoding with 'input_ids' to create a padding mask.
        pad_token_id: The token ID used for padding.
        median: If True, returns the median loss per sequence; otherwise, returns the mean.
        sample_p: If True, sample target tokens from p_logits; otherwise, use argmax.
        temperature: Temperature scaling factor applied to logits.

    Returns:
        A NumPy array of aggregated cross-entropy values (one per sequence).
    """
    vocab_size = p_logits.shape[-1]
    seq_length = q_logits.shape[-2]
    p_scores = p_logits / temperature
    q_scores = q_logits / temperature

    p_proba = softmax_fn(p_scores)  # shape: [batch_size, seq_len, vocab_size]

    if sample_p:
        target_tokens = torch.multinomial(
            p_proba.view(-1, vocab_size), num_samples=1, replacement=True
        ).view(p_logits.shape[0], seq_length)
    else:
        target_tokens = p_proba.argmax(dim=-1)

    ce_losses = ce_loss_fn(q_scores.transpose(1, 2), target_tokens)
    ce_losses = ce_losses.view(p_logits.shape[0], seq_length)

    padding_mask = encoding.input_ids != pad_token_id

    ce_losses = ce_losses.masked_fill(~padding_mask, float("nan"))

    if median:
        agg_ce = np.nanmedian(ce_losses.cpu().float().numpy(), axis=1)
    else:
        agg_ce = (
            (ce_losses * padding_mask.float()).sum(1)
            / padding_mask.float().sum(1)
        ).to("cpu").float().numpy()

    return agg_ce
