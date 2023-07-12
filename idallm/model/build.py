import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

'''
TODO: 
    - Add adapter support
'''

def init_causallm_acc(model_dir, tokenizer_dir=None, **kwargs):
    if tokenizer_dir is None: tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    config = AutoConfig.from_pretrained(model_dir, **kwargs)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(config)

    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, model_dir, device_map="auto"
    )

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer

def init_causallm(model_dir, tokenizer_dir=None, **kwargs):
    if tokenizer_dir is None: tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).cuda()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

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

