from typing import Any
from vllm.sampling_params import SamplingParams
from typing import List, Union
import json

from yaml import load, save
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_yaml(path : str) -> dict:
    return load(open(path), Loader=Loader)

def parse_input(input : str) -> Union[List[str], List[dict]]:
    if 'json' in input:
        type = 'json'
        return [json.loads(line) for line in open(input)], type
    elif 'txt' in input:
        type = 'txt'
        return [line.strip() for line in open(input)], type   
    elif 'yaml' in input:
        type = 'yaml'
        return load_yaml(input), type
    else:
        raise ValueError(f"Unsupported input format: {input}")
    
def save_output(output : List[str], out_file : str, type : str) -> None:
    if type == 'json':
        with open(out_file, 'w') as f:
            for line in output: f.write(json.dumps(line) + '\n')
    elif type == 'txt':
        with open(out_file, 'w') as f:
            for line in output: f.write(line + '\n')
    elif type == 'yaml':
        with open(out_file, 'w') as f: f.write(save(output, f))
    else: raise ValueError(f"Unsupported output format: {out_file}")
    
class LLM(object):
    def __init__(self, engine, generation_config) -> None:
        self.engine = engine
        generation_config = load_yaml(generation_config)
        self.include_prompt = generation_config.pop("include_prompt", False)
        include_logits = generation_config.pop("include_logits", None)
        include_prompt_logits = generation_config.pop("include_prompt_logits", None)

        prompt_json = generation_config.pop("prompt", None)
        if prompt_json is not None: 
            from lightchain import Prompt
            self.prompt = Prompt.from_json(prompt_json)
        else: self.prompt = None

        self.generation_config = SamplingParams(**generation_config, logprobs=include_logits, prompt_logprobs=include_prompt_logits)
    
    def __call__(self, prompts) -> Any:
        if isinstance(prompts, str): texts = [prompts]
        if self.prompt is not None: prompts = map(self.prompt, prompts)
        else: pass
        request_id = 0
        while prompts or self.engine.has_unfinished_requests():
            if prompts:
                self.engine.add_request(str(request_id), prompts.pop(0), self.generation_config)
            request_id += 1

        request_outputs = self.engine.step()

        outputs = [output for output in request_outputs if output.finished]
        texts = [output.text for output in outputs]
        if self.include_prompt: texts = [prompt + text for prompt, text in zip(prompts, texts)]
        out = {'text': texts}
        if self.include_logits: out['logits'] = [output.logprobs for output in outputs]
        if self.include_prompt_logits: out['prompt_logits'] = [output.prompt_logprobs for output in outputs]
        return out