#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch


# Config that serves all environment
GLOBAL_CONFIG = {
    "USE_CUDA_IF_AVAILABLE": True,
    "ROUND_DIGIT": 6,
    "BATCH_SIZE": 8,
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
        "ROUND_DIGIT": 3
    }
}


def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV
    config['MODEL_TYPE'] = os.getenv('MODEL_TYPE', 'causallm')
    config['REMOVE_PROMPT'] = int(os.getenv('REMOVE_PROMPT', 0))
    config['MODEL_NAME_OR_PATH'] = os.getenv('MODEL_NAME_OR_PATH')
    config['DISTRIBUTED'] = int(os.getenv('DISTRIBUTED', 0))
    config['QUANTIZED'] = int(os.getenv('QUANTIZED', 0))
    config['MAX_INPUT_LENGTH'] = int(os.getenv('MAX_INPUT_LENGTH', 512))
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() and config['USE_CUDA_IF_AVAILABLE'] else 'cpu'
    config['DEBUG'] = int(os.getenv('DEBUG', 0))

    return config

# load config for import
CONFIG = get_config()

if __name__ == '__main__':
    # for debugging
    import json
    print(json.dumps(CONFIG, indent=4))