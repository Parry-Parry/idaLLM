# LLM Utility for the IDA Cluster

## Install
```
pip install --upgrade git+https://github.com/Parry-Parry/idaLLM.git
```

## Functions

* LLM Serving: Through vLLM and FastAPI you can run an efficient endpoint for LLM inference
* Local Inference: Through vLLM you can execute prompts (coupled with LightChain)

## Requires:
* torch : self-explanatory
* transformers : self-explanatory
* fastapi: Backbone used to serve applications over TCP
* ray: Deployment component
* vllm: Efficient model serving
## Optional
* LightChain: Makes life a lot easier in terms of prompt formatting, chain-prompting and chat functionality
