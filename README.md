# LLM Utility for the IDA Cluster

## Install
```
pip install --upgrade git+https://github.com/Parry-Parry/idaLLM.git
```

## Functions

* LLM Serving: Through vLLM and FastAPI you can run an efficient endpoint for LLM inference
* Local Inference: Through vLLM you can execute prompts (coupled with LightChain)

## How-To

To run the API in its simplest form
```
python -m idallm.api.serve --model <MODEL_ID> --host 0.0.0.0 --port 8080
```

To run local inference in its simplest form (over a text file )
```
python -m idallm.local.serve --model <MODEL_ID> --input_file my_prompts.txt --output_file my_outputs.txt
```

## Requires:
* torch : self-explanatory
* transformers : self-explanatory
* fastapi: Backbone used to serve applications over TCP
* ray: Deployment component
* vllm: Efficient model serving
## Optional:
* LightChain: Makes life a lot easier in terms of prompt formatting, chain-prompting and chat functionality
