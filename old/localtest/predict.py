from typing import List, Tuple
import torch
import numpy as np
import logging

from .config import CONFIG
from .util import batch, cut_prompt

def preprocess(package: dict, text : List[str]) -> list:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding
    :param package: dict from fastapi state including model and preocessing objects
    :param text: list of input to be proprocessed
    :return: tensor of preprocessed input
    """

    # preprocessing text for sequence_classification, token_classification or text_generation

    inputs = package["tokenizer"].encode_plus(
        text,
        max_length=CONFIG['MAX_INPUT_LENGTH'],
        padding='longest',
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(package["model"].device)

    return input_ids

def iterate_batch(X, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:min(i + batch_size, len(X))]

def predict(package: dict, text : List[str], generation_params : dict) -> Tuple[str, np.ndarray]:
    """
    Run model and get result
    :param package: dict from fastapi state including model and preocessing objects
    :param text: list of input values
    :param generation_params: dict of generation parameters
    :return: text, numpy array of model output
    """

    # process data
    logging.info('Preprocessing data')

    # run model
    logging.info('Running model')

    logging.info('Generating text')
    texts = ['xyz test'] * len(text)
    logits = torch.randn((len(text), 46)).numpy()
    logging.info('Decoding text')
    return texts, logits