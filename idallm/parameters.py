'''
Full parameter sets can be found in the vllm documentation.
Refer to: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
'''

BEAM_SEARCH = {
    'use_beam_search' : True,
    'temperature' : 0.,
    'best_of' : 5,
    'top_p' : 0.75,
    'early_stopping' : True,
}

GREEDY = {
    'temperature': 0.,
}

def load_generation_config(type, **kwargs):
    types = {
    "beam_search":BEAM_SEARCH,
    "greedy": GREEDY,
    }
    return {**types[type], **kwargs}