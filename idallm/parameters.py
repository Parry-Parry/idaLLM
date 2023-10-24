'''
Full parameter sets can be found in the vllm documentation.
Refer to: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
'''

BEAM_SEARCH = {
    'use_beam_search' : True,
    'n' : 5,
    'top_p' : 0.75,
    'early_stopping' : True,
}
    
GREEDY_SMOOTH = {
    'do_sample': False,
    'temperature': 2.0,
    'top_p' : 0.75,
}

GREEDY_DETERMINISTIC = {
    'do_sample': False,
    'temperature': 0.1,
    'top_p' : 0.75,
}

def load_generation_config(type, **kwargs):
    types = {
    "beam_search":BEAM_SEARCH,
    "greedy_smooth": GREEDY_SMOOTH,
    "greedy_deterministic": GREEDY_DETERMINISTIC,
    }
    return {**types[type], **kwargs}