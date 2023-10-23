import itertools

def cut_prompt(output : str, input : str):
    return output[len(input):]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatenate(*lists):
    return [*itertools.chain.from_iterable(lists)]