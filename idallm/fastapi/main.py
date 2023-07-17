#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from fire import Fire

import uvicorn
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

from idallm.model.build import init_causallm_acc, init_causallm, init_8bitcausallm
from idallm.fastapi.predict import predict
from idallm.fastapi.config import CONFIG
from idallm.fastapi.schema import *

# Initialize API Server
app = FastAPI(
    title="LLM API",
    description="Deployed inference on Llama",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
#app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
#app.add_exception_handler(RequestValidationError, validation_exception_handler)
#app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model

    model, tokenizer = init_causallm()

    # add model and other preprocess tools to app state
    app.package = {
        "model": model,
        "tokenizer" : tokenizer
    }


@app.post('/generate',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')

    # prepare input data
    prompts = body.text
    generation_params = body.generation_params

    # run model inference
    text, logits = predict(app.package, prompts, generation_params)

    # round probablities for json
    logits = np.around(logits, decimals=CONFIG['ROUND_DIGIT']).tolist()

    # prepare json for returning
    results = {
        'text': text,
        'logits': logits
    }

    return {
        "error": False,
        "results": results
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

def main(host : str, port : int):
    uvicorn.run("main:app", host=host, port=port,
                reload=True,
                )

if __name__ == '__main__':
    Fire(main)