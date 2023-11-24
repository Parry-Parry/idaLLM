import argparse
import json
import logging

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from . import LLM, parse_input, save_output

engine = None

if __name__ == "__main__":
    """
     Args:
        model: str
        tokenizer: Optional[str] = None
        tokenizer_mode: str = 'auto'
        trust_remote_code: bool = False
        download_dir: Optional[str] = None
        load_format: str = 'auto'
        dtype: str = 'auto'
        seed: int = 0
        max_model_len: Optional[int] = None
        worker_use_ray: bool = False
        pipeline_parallel_size: int = 1
        tensor_parallel_size: int = 1
        max_parallel_loading_workers: Optional[int] = None
        block_size: int = 16
        swap_space: int = 4  # GiB
        gpu_memory_utilization: float = 0.90
        max_num_batched_tokens: Optional[int] = None
        max_num_seqs: int = 256
        max_paddings: int = 256
        disable_log_stats: bool = False
        revision: Optional[str] = None
        tokenizer_revision: Optional[str] = None
        quantization: Optional[str] = None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--generation_config", type=str, default=None)
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    input_file = args.input_file
    out_file = args.out_file
    generation_config = args.generation_config

    delattr(args, "input_file")
    delattr(args, "out_file")
    delattr(args, "generation_config")

    logging.basicConfig(level=logging.INFO)

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    model = LLM(engine, generation_config)

    texts, type = parse_input(input_file)
    output = model(texts)
    model.save_output(output, out_file, type)