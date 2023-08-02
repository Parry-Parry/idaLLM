from typing import List, Tuple
import torch
import numpy as np

from idallm.fastapi.config import CONFIG
from idallm.fastapi.util import batch, cut_prompt

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

    input_ids = inputs["input_ids"].to(0)

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
    :return: numpy array of model output
    """

    # process data
    X = preprocess(package, text)

    # run model
    model = package['model']
    if len(X) > CONFIG['BATCH_SIZE']:
        X_batches = list(batch(X, CONFIG['BATCH_SIZE']))
        outputs = []
        for X_batch in X_batches:
            with torch.no_grad():    
                outputs_batch = package['model'].generate(
                    X_batch, output_scores=True, return_dict_in_generate=True, **generation_params
                )
            outputs.append(outputs_batch)
        sequences = torch.cat([output.sequences.cpu() for output in outputs], dim=0).numpy()
        logits = torch.cat([torch.cat(list(output.scores), dim=0).cpu() for output in outputs], dim=0).numpy()
    else:
        with torch.no_grad():    
            outputs = model.generate(
                X, output_scores=True, return_dict_in_generate=True, **generation_params
            )
            sequences = outputs.sequences.cpu().numpy()
            logits = torch.cat(list(outputs.scores), dim=0).cpu().numpy()
    
    texts = package["tokenizer"].batch_decode(sequences, skip_special_tokens=True)
    if CONFIG['REMOVE_PROMPT']: texts = list(map(cut_prompt, texts, text))
    return texts, logits