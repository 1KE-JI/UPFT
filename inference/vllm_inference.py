import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, GenerationConfig
from transformers import set_seed

from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import torch
import asyncio

from transformers.utils import logging

from typing import Union, Any
import random
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from chat_template import CHAT_TEMPLATE
from utils.utils import *
from utils.rollout_utils import *

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)
offline = False


def run_rollout(args):
    add_prompt = args.add_prompt
    if "Qwen2.5-Math" in args.model_path or "DeepSeek" in args.model_path:
        add_prompt = ""
    data_ratio = args.data_ratio
    data_path, _ = get_data_path(dataset_name=args.dataset_name, data_ratio=data_ratio)

    parts = args.model_path.split("/")
    args.model_name = "/".join(parts[-2:])

    try:
        with open(data_path) as fp:
            datasets = [json.loads(line) for line in list(fp)]
    except Exception:
        with open(data_path) as fp:
            datasets = json.load(fp)
        for data in datasets:
            if isinstance(data['response'], list):
                data['response'] = data['response'][0]
    print("data_path:", data_path)
    if "id" not in datasets[0]:
        for idx, data in enumerate(datasets):
            data['id'] = idx
        write_jsonl(data_path, datasets)
    print("Examples:", datasets[0])
    print("model_path:", args.model_path)
    print("model_name:", args.model_name)

    # init tokenizer and chat template
    left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        chat_template_name = "llama-3.1-chat"
    elif args.model_name == "meta-llama/Llama-3.2-1B-Instruct":
        chat_template_name = "llama-3.2-chat"
    elif args.model_name == "meta-llama/Llama-3.2-3B-Instruct":
        chat_template_name = "llama-3.2-chat"
    elif args.model_name == "meta-llama/Llama-3.3-70B-Instruct":
        chat_template_name = "llama-3.3-chat"
    elif args.model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":
        chat_template_name = "DeepSeek-R1-Distill-Llama-70B"
    elif args.model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        chat_template_name = "DeepSeek-R1-Distill-Llama-8B"
    elif args.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        chat_template_name = "DeepSeek-R1-Distill-Qwen-7B"
    elif args.model_name == "Qwen/Qwen2.5-Math-7B-Instruct":
        chat_template_name = "Qwen2.5-Math-7B-Instruct"
    elif args.model_name == "Qwen/Qwen2.5-Math-72B-Instruct":
        chat_template_name = "Qwen2.5-Math-72B-Instruct"
    elif args.model_name == "Qwen/Qwen2.5-7B-Instruct":
        chat_template_name = "Qwen2.5-7B-Instruct"
    elif args.model_name == "Qwen/Qwen2.5-3B":
        chat_template_name = "Qwen2.5-3B"
    else:
        raise NotImplementedError
    print("Using chat template:", chat_template_name)
    left_tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    # create dataset
    source_path_ids = f"random_ids_{args.max_samples}_rollout_tokenized_1W_length_{len(datasets)}.json"
    print("source_path_ids:", source_path_ids)
    print("testset length:", len(datasets))
    print("args:", args)

    tokenized_prompt_token_ids = []
    print_flag = False
    if "prompt_token_ids" not in datasets[0]:
        for data in tqdm(datasets) or True:
            if "problem" not in data and "question" in data:
                data["problem"] = data["question"]
            elif 'problem' not in data and 'query' in data:
                data['problem'] = data['query']
            else:
                raise NotImplementedError

            messages = [{"role": "user", "content": "" + data['problem'] + add_prompt}]

            prompt_template = left_tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tokenized_prompt = left_tokenizer.tokenize(prompt_template)
            if args.prefix_length != 0:
                if "answer" in data:
                    response = data['answer']
                elif "response" in data:
                    response = data['response']
                else:
                    raise NotImplementedError
                tokenized_response = left_tokenizer.tokenize(response)
                tokenized_response = tokenized_response[:args.prefix_length]
                tokenized_prompt.extend(tokenized_response)
            tokenized_prompt_token_id = left_tokenizer.convert_tokens_to_ids(tokenized_prompt)
            data["tokenized_prompt"] = tokenized_prompt
            data["prompt_token_ids"] = tokenized_prompt_token_id
            tokenized_prompt_token_ids.append(tokenized_prompt_token_id)

            if not print_flag:
                print("data:", data)
                print("tokenized_prompt:", " ".join(tokenized_prompt))
                print_flag = True
    if args.max_samples != -1:
        print("max_samples:", args.max_samples)

        if not os.path.exists(source_path_ids):
            print("random_ids.json not found:", source_path_ids)
            random_ids = random.sample(range(len(datasets)), args.max_samples)
            with open(source_path_ids, 'w') as fp:
                json.dump(random_ids, fp)
        else:
            print("random_ids.json found:", source_path_ids)
            with open(source_path_ids) as fp:
                random_ids = json.load(fp)
                print("# random_ids.json:", len(random_ids))

        new_datasets = []

        for random_id in random_ids:
            data = datasets[random_id]
            data['id'] = random_id
            new_datasets.append(data)
        datasets = new_datasets
    else:
        print("max_samples = -1")
    print("len(datasets):", len(datasets))

    model_type = args.model_name.split('/')[-1]
    if args.prefix_length == 0:
        source_path = f"{model_type}_{args.dataset_name}_data_length_{len(datasets)}_max{args.max_tokens}_n_{args.n}.json"
        tgt_path = f"tgt_{model_type}_{args.dataset_name}_data_length_{len(datasets)}_max{args.max_tokens}_n_{args.n}.json"
    else:
        source_path = f"{model_type}_{args.dataset_name}_data_length_{len(datasets)}_max{args.max_tokens}_n_{args.n}_prefix_length_{args.prefix_length}_noposterior.json"
        tgt_path = f"tgt_{model_type}_{args.dataset_name}_data_length_{len(datasets)}_max{args.max_tokens}_n_{args.n}_prefix_length_{args.prefix_length}_noposterior.json"

    source_base_path = "rollout_result_source"
    tgt_base_path = "rollout_result_tgt"
    if not os.path.exists(source_base_path):
        os.makedirs(source_base_path)
    if not os.path.exists(tgt_base_path):
        os.makedirs(tgt_base_path)
    source_path = os.path.join(source_base_path, source_path)
    tgt_path = os.path.join(tgt_base_path, tgt_path)

    keep_keys = ['question', 'source_type', 'metadata', 'id', 'problem', 'response']

    new_datasets = []
    for data in tqdm(datasets):
        for i in range(args.n):
            if i == 0:
                _data = dict()
                for k, v in data.items():
                    _data[k] = v
            else:
                _data = dict()
                _data['query'] = data["query"]
                _data["prompt_token_ids"] = data["prompt_token_ids"]

            _data["id"] = len(new_datasets)
            new_datasets.append(_data)

    write_jsonl(source_path, new_datasets)
    print("source_path:", source_path)
    print("tgt_path:", tgt_path)
    print("len(new_datasets):", len(new_datasets))
    print("tokenized_prompt:", new_datasets[0]["tokenized_prompt"])

    generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print(generation_config)
    sampling_kwargs = {
        "top_p": generation_config.top_p,
        "top_k": -1 if generation_config.top_k == 0 else generation_config.top_k,
        "temperature": 0.6,
        "max_tokens": args.max_tokens,
        "n": 1
    }

    tensor_parallel_size = 8
    if "DeepSeek" in args.model_path:
        tensor_parallel_size = 4
    if "Qwen" in args.model_path:
        tensor_parallel_size = 4

    print(json.dumps(sampling_kwargs, indent=4))
    sampling_params = SamplingParams(**sampling_kwargs)
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype='bfloat16',
        seed=args.seed,
        num_scheduler_steps=1,
        trust_remote_code=True,
    )
    outputs = model.generate(prompt_token_ids=tokenized_prompt_token_ids, sampling_params=sampling_params,
                             use_tqdm=True)
    generations = []

    for data, o, pt_ids in zip(datasets, outputs, tokenized_prompt_token_ids):
        if "solution" in data:
            data["ori_response"] = data["solution"]
        else:
            data["ori_response"] = data["response"]
        data["response"] = [r.text for r in o.outputs]
        data["prompt_token_ids"] = pt_ids
        if args.dataset_name == "s1":
            for k in data:
                if k not in keep_keys:
                    del data[k]
        generations.append(data)
    write_jsonl(tgt_path, generations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--dataset_name',
                        default="limo",
                        type=str)
    parser.add_argument('--model_path',
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        type=str)
    parser.add_argument('--model_name', default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument('--bf16',
                        default=True,
                        type=bool)
    parser.add_argument('--add_prompt',
                        # default=" Please wrap the final answer in $\\boxed{}$ tag. Let's think step by step.",
                        default="",
                        type=str)
    parser.add_argument('--data_ratio',
                        default=-1,
                        type=float)
    parser.add_argument('--max_samples',
                        default=-1,
                        type=int)
    parser.add_argument('--prefix_length',
                        default=0,
                        type=int)
    parser.add_argument('--n',
                        default=1,
                        type=int)
    parser.add_argument('--task',
                        default="rollout",
                        type=str)
    parser.add_argument('--max_tokens',
                        default=16384,
                        type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    print(args)

    task = args.task
    if task == "rollout":
        prefix_length_list = [0]
        for prefix_length in prefix_length_list:
            print("processing prefix_length:", prefix_length)
            args.prefix_length = prefix_length
            run_rollout(args)
    else:
        raise NotImplementedError
