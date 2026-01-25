import os
from nanovllm import LLM, SamplingParams
from nanovllm.utils.logger import logger
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/data/xbw/models/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "写一个简单的 python 函数，实现计算两个数的和",
        "我有一个狗，它",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info("\n")
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
