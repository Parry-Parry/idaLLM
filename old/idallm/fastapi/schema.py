
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    text : List[str] = Field(..., example=['I am a Language Model'], title='Prompt for generation')
    generation_params : Dict[str, Any] = Field(..., example={'max_length': 50}, title='Generation parameters')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    text : list = Field(..., example=['I am a Language Model'], title='Returned Text')
    logits : list = Field(..., example=[[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]], title='Log Probabilities over generation')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')
