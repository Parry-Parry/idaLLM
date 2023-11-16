import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
GLOBALMAX_TOKENS = 512
app = FastAPI()
engine = None
tokenizer = None

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    if isinstance(prompt, str):
        prompt = [prompt]
    
    prompt = [p for p in prompt if len(tokenizer.encode(p)) < GLOBALMAX_TOKENS]

    stream = request_dict.pop("stream", False)
    include_logits = request_dict.pop("include_logits", None)
    sampling_params = SamplingParams(**request_dict, logprobs=include_logits)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)
    
    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Using background_taks to abort the the request
        # if the client disconnects.
        background_tasks.add_task(may_abort_request, request_id)
        return StreamingResponse(
            stream_results(results_generator), background=background_tasks
        )

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    if include_prompt: 
            prompt = final_output.prompt
            text_outputs = [prompt + output.text for output in final_output.outputs]
    else:
        text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    if include_logits:
        logits = [output.logprobs for output in final_output.outputs]
        logits = [[{k: round(v, 4) for k, v in logit.items()} for logit in logprobs] for logprobs in logits] 
        ret["logits"] = logits
    return JSONResponse(ret)


if __name__ == "__main__":
    """
    Args:
        model: name or path of the huggingface model to use
        download_dir: directory to download and load the weights,
            default to the default cache dir of huggingface.
        use_np_weights: save a numpy copy of model weights for
            faster loading. This can increase the disk usage by up to 2x.
        use_dummy_weights: use dummy values for model weights.
        dtype: data type for model weights and activations.
            The "auto" option will use FP16 precision
            for FP32 and FP16 models, and BF16 precision.
            for BF16 models.
        seed: random seed.
        worker_use_ray: use Ray for distributed serving, will be
            automatically set when using more than 1 GPU
        pipeline_parallel_size: number of pipeline stages.
        tensor_parallel_size: number of tensor parallel replicas.
        block_size: token block size.
        swap_space: CPU swap space size (GiB) per GPU.
        gpu_memory_utilization: the percentage of GPU memory to be used for
            the model executor
        max_num_batched_tokens: maximum number of batched tokens per iteration
        max_num_seqs: maximum number of sequences per iteration.
        disable_log_stats: disable logging statistics.
        engine_use_ray: use Ray to start the LLM engine in a separate
            process as the server process.
        disable_log_requests: disable logging requests.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)