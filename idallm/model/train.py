from functools import partial
from typing import List
from terrierutil.llm.model.build import init_causallm, generate_and_tokenize_prompt
from ..util import LlamaConfig, LoraConfig
from ..util.modelconfig import load_model_config

from datasets import load_dataset
import transformers
from lightchain.prompt import Prompt

import fire
import os
import torch

def main(datasuffix : str, 
         datadir : str,
         outputdir : str,
         prompt_file : str, 
         model_name : str = "7B",
         batch_size : int = 128,
         micro_batch_size : int = 4,
         lr : float = 3e-4,
         epochs : int = 3,
         resume_from_checkpoint : str = None,
         validation_split : float = None,
         train_on_inputs : bool = False,
         add_eos_token : bool = True,
         cutoff_len : int = 1024,
         lora_r: int = 8,
         lora_alpha: int = 16,
         lora_dropout: float = 0.05,
         lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
         ]
         ):
    
    '''
    Collect Variables and Load Data
    --------------------------------
    '''
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    data_files = {"train": f"train.{datasuffix}", "test": f"test.{datasuffix}"}
    if datasuffix=='json' or datasuffix=='jsonl':
        data = load_dataset("json", data_files=datadir)
    else:
        data = load_dataset(datadir, data_files=data_files)

    prompt = Prompt.fromjson(prompt_file)

    '''
    Intialize Model
    --------------------------------
    '''

    loracfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    cfg = LlamaConfig.build_llama_config(load_model_config(model_name))
    model, tokenizer = init_llama(cfg, loracfg, resume_from_checkpoint=resume_from_checkpoint)

    '''
    Data Splits and Preprocessing
    --------------------------------
    '''

    gentok = partial(generate_and_tokenize_prompt, prompt, tokenizer, train_on_inputs, add_eos_token, cutoff_len)
    if validation_split > 0:
        train_val = data["train"].train_test_split(
            test_size=validation_split, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(gentok)
        )
        val_data = (
            train_val["test"].shuffle().map(gentok)
        )
    else:
        train_data = data["train"].shuffle().map(gentok)
        val_data = None

    '''
    Training
    --------------------------------
    '''

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_data > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_data > 0 else None,
            save_steps=200,
            output_dir=outputdir,
            save_total_limit=3,
            load_best_model_at_end=True if val_data > 0 else False,
            group_by_length=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(outputdir)

if __name__ == '__main__':
    fire.Fire(main)