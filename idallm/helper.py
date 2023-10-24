from vllm import LLM

def model_init(model_name_or_path, dtype : str = None, quantization : str = None, **kwargs):
    model = LLM(model=model_name_or_path, dtype=dtype, quantization=quantization, **kwargs)
    return model, model.get_tokenizer()