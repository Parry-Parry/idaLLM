import sys
import torch
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from idallm.fastapi.config import CONFIG

def get_map(model_id : str, mem : dict, do_int8 : bool = True):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_id)
        model = LlamaForCausalLM(config)
    
    device_map = infer_auto_device_map(
        model, max_memory=mem, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"]
    )
    del model 
    return device_map

def get_mem(ngpu : int, gpu_type : str ='3090', cpu_mem : int = 0) -> dict:
    types = {
        '3090' : 20,
        'titan' : 20,
        'a6000' : 40
    }
    if ngpu == 1: return {0 : f'{types[gpu_type]}GIB'}
    mapping = {0 : f'{types[gpu_type]-4}GIB'}
    for i in range(1, ngpu):
        mapping[i] = f'{types[gpu_type]}GiB'
    if cpu_mem != 0: mapping['cpu'] = f'{cpu_mem}GIB'
    return mapping

'''
TODO: 
    - Add adapter support
'''

def init_causallm(**kwargs):
    model_dir = CONFIG["MODEL_NAME_OR_PATH"]
    tokenizer_dir = model_dir
    additional_kwargs = {}

    if CONFIG["QUANTIZED"]:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"],
            llm_int8_enable_fp32_cpu_offload=True
        )
        additional_kwargs.update({
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "device_map": "auto"
        })

    if CONFIG["DISTRIBUTED"]:
        additional_kwargs.update({
            "device_map": "auto",
        })

    model = AutoModelForCausalLM.from_pretrained(model_dir, **additional_kwargs, **kwargs)
    if CONFIG['HALF'] and 'device_map' not in additional_kwargs: model = model.half()
    if 'device_map' not in additional_kwargs: model = model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference

    model.eval()
    
    return model, tokenizer

def tokenize(prompt, tokenizer, add_eos_token=True, cutoff_len=1024):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

def generate_and_tokenize_prompt(prompt, 
                                 tokenizer,
                                 train_on_inputs, 
                                 add_eos_token,
                                 cutoff_len,
                                 data_point):
        full_prompt = prompt.construct(**data_point)
        tokenized_full_prompt = tokenize(full_prompt, tokenizer=tokenizer, add_eos_token=add_eos_token, cutoff_len=cutoff_len)
        if not train_on_inputs:
            data_point.pop("output")
            user_prompt = prompt.construct(**data_point)
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  
        return tokenized_full_prompt
