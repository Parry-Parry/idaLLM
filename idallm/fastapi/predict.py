from typing import List, Tuple
import torch
import numpy as np

from idallm.fastapi.config import CONFIG

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
        max_length=CONFIG['max_input_length'],
        padding='longest',
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(0)

    return input_ids

def iterate_batch(X, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:min(i + batch_size, len(X))]

def predict(package: dict, text : str, generation_params : dict) -> Tuple[str, np.ndarray]:
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
        X_batches = list(iterate_batch(X, CONFIG['BATCH_SIZE']))
        outputs = []
        for X_batch in X_batches:
            with torch.no_grad():    
                outputs_batch = package['model'].generate(
                    X_batch, **generation_params
                ).cpu()
            outputs.append(outputs_batch)
        outputs = torch.cat(outputs)
    else:
        with torch.no_grad():    
            outputs = model.generate(
                X, **generation_params
            ).cpu()

    texts = package["tokenizer"].batch_decode(outputs, skip_special_tokens=True)
    logits = outputs.logits.numpy()

    return texts, logits