import torch
import sys
import os

from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset, load_from_disk
from typing import Literal
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.openmathinst_utils import process_results, extract_answer
from inference.chat_template import CHAT_TEMPLATE
from utils.data_utils import write_jsonl, most_common_element

from utils.openmathinst_utils import extract_answer
from utils.openmathinst_utils import math_equal

import json
import argparse

DATASET_ID = "zwhe99/MATH"


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def read_jsonl(file):
    with open(file, "r") as f:
        return [json.loads(l) for l in tqdm(f, desc="loading jsonl file: {}".format(file))]


def compute_acc(gens):
    correct = sum([g["correct"] for g in gens])
    return correct / len(gens)


def _judge(llm_judge, generations, generation_file, result_file, maj_vote=1, bf16=False, fp16=False, seed=42,
           enforce_eager=False, num_scheduler_steps=1, add_rule_correctness=True, args=None):
    # compute correctness
    JUDGE_LLM_PATH = args.judge_llm_path
    if not llm_judge:
        # This is to allow two output formats
        # MetaMathQA: `The answer is: 123`
        # OpenMathInstruct: `\boxed{123}`
        for g in tqdm(generations, desc="computing correctness", total=len(generations)):
            g["correct"] = (
                    process_results(
                        g["response"],
                        g["solution"],
                        response_extract_from_boxed=True,
                        answer_extract_from_boxed=True,
                    ) or
                    process_results(
                        g["response"],
                        g["solution"],
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                        answer_extract_from_boxed=True,
                    )
            )

        # save generations
        write_jsonl(generation_file, generations)

        # fix the format of generations
        for g in generations:
            if "type" not in g:
                g["type"] = "unknown"
            if "level" not in g:
                g["level"] = "unknown"
        # save evaluation results
        all_types = sorted(list(set([g["type"] for g in generations])))
        all_levels = sorted(list(set([g["level"] for g in generations])))

        with open(result_file, "w") as f:
            for t in all_types:
                gens = [g for g in generations if g["type"] == t]
                f.write(f"{t}: {compute_acc(gens) * 100:.1f}\n")
            for l in all_levels:
                gens = [g for g in generations if g["level"] == l]
                f.write(f"{l}: {compute_acc(gens) * 100:.1f}\n")
            f.write(f"Overall: {compute_acc(generations) * 100:.1f}\n")
    else:
        # compute correctness
        dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        print(f"LLM Judge: {JUDGE_LLM_PATH}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"dtype: {dtype}")
        print(f"seed: {seed}")
        print(f"enforce_eager: {enforce_eager}")
        print(f"num_scheduler_steps: {num_scheduler_steps}")
        judge_llm = LLM(
            model=JUDGE_LLM_PATH,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=dtype,
            seed=seed,
            max_model_len=16 * 1024,
            enforce_eager=enforce_eager,
            num_scheduler_steps=num_scheduler_steps,
            trust_remote_code=True,
            distributed_executor_backend="mp"
        )
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_LLM_PATH, trust_remote_code=True)

        # flatten all generations
        # judge_prompts = [tokenizer.get_context(g["problem"], extract_answer(g["solution"]), r) for g in generations for r in g["response"]]
        for g in generations:
            # pre
            if "answer" not in g and "Answer" in g:
                g["answer"] = g["Answer"]
            if "answer" not in g and "final_answer" in g:
                if isinstance(g["final_answer"], list):
                    g["final_answer"] = g["final_answer"][0]
                g["answer"] = g["final_answer"]
            if "solution" in g:
                if isinstance(g["solution"], list):
                    g["solution"] = g["solution"][0]
            # do
            if "extract_answer" not in g:
                if "gsm8k" in generation_file:
                    g["extract_answer"] = g['answer'].split()[-1]
                elif "omni_math" in generation_file:
                    g["extract_answer"] = g['answer']
                else:
                    if "solution" in g and "boxed" in g["solution"]:
                        g["extract_answer"] = extract_answer(g["solution"])
                    else:
                        g["extract_answer"] = g["answer"]
            if "aime" in generation_file:
                g["extract_answer"] = g["answer"]

        judge_prompts = [tokenizer.get_context(g["problem"], g["extract_answer"], r) for g in generations for r in
                         g["response"]]

        judge_outputs = judge_llm.generate(
            judge_prompts,
            sampling_params=SamplingParams(
                temperature=0,
                max_tokens=300,
                stop_token_ids=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            ),
        )

        # re-group judge outputs
        assert len(judge_outputs) % maj_vote == 0 and len(judge_outputs) // maj_vote == len(
            generations), "Number of judge outputs should be a multiple of maj_vote"
        grouped_judge_outputs = [judge_outputs[i: i + maj_vote] for i in range(0, len(judge_outputs), maj_vote)]

        for g, gjo in zip(generations, grouped_judge_outputs):
            g["judge_res"] = []
            for jo in gjo:
                try:
                    judge_res = tokenizer.parse_response(jo.outputs[0].text)
                except Exception as e:
                    judge_res = "Failed to parse"
                g["judge_res"].append(judge_res)

            all_answers = [jr["answer"] for jr in g["judge_res"] if jr != "Failed to parse"]
            all_correctness = [jr["judgement"] == "TRUE" for jr in g["judge_res"] if jr != "Failed to parse"]
            assert len(all_answers) == len(
                all_correctness), "Number of answers and correctness should be one-to-one mapping"

            most_common_answer = most_common_element(all_answers)
            g["correct"] = all_correctness[all_answers.index(most_common_answer)]

        if add_rule_correctness:
            for generation in tqdm(generations):
                generation["rule_correctness"] = math_equal(extract_answer(generation["response"][0]),
                                                            generation["extract_answer"])
            hit = 0
            for generation in generations:
                # raw numical \\boxed result
                generation["llm_correctness"] = generation["correct"]
                correct = generation["llm_correctness"]
                pred = extract_answer(generation["response"][0])

                if generation["rule_correctness"] or generation["llm_correctness"]:
                    correct = True
                if generation["rule_correctness"] and not generation["llm_correctness"] and correct:
                    hit += 1
                generation["correct"] = correct
            print("hit:", hit)
        # save generations
        write_jsonl(generation_file, generations)
        print("dataset_path:", generation_file)
        # fix the format of generations
        for g in generations:
            if "type" not in g:
                g["type"] = "unknown"
            if "level" not in g:
                g["level"] = "unknown"
            if g['level'] is None:
                g["level"] = "unknown"
        # save evaluation results
        # print("generations[0]:", generations[0])
        all_types = sorted(list(set([g["type"] for g in generations])))
        all_levels = sorted(list(set([g["level"] for g in generations])))

        with open(result_file, "w") as f:
            for t in all_types:
                gens = [g for g in generations if g["type"] == t]
                f.write(f"{t}: {compute_acc(gens) * 100:.1f}\n")
            for l in all_levels:
                gens = [g for g in generations if g["level"] == l]
                f.write(f"{l}: {compute_acc(gens) * 100:.1f}\n")
            f.write(f"Overall: {compute_acc(generations) * 100:.1f}\n")
        from pathlib import Path
        path = Path(generation_file)
        dataset_name = path.parent.__str__().split("/")[-1]
        cur_dir = path.parent.parent
        if os.path.exists(os.path.join(cur_dir, "eval_results.json")):
            with open(os.path.join(cur_dir, "eval_results.json"), "r") as f:
                eval_results = json.load(f)
        else:
            eval_results = {}
        eval_results[dataset_name] = compute_acc(generations) * 100
        with open(os.path.join(cur_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
    return generations


def eval(
        # required
        base_model: str = None,
        chat_template_name: str = None,
        output_dir: str = None,

        # model
        bf16: bool = False,
        fp16: bool = False,
        peft_path: str = None,

        # data
        data_dir: str = None,  # If provided, the data will loaded from data_dir/DATASET_ID
        split: Literal["test", "math500", "math4500"] = "test",
        data_file: str = None,  # If provided, the data will loaded from data_file

        # gen
        enforce_eager: bool = False,
        num_scheduler_steps: int = 1,
        maj_vote: int = 1,
        add_prompt: str = None,
        add_prompt_prefix: str = None,
        max_model_len: int = 8192,
        max_generation_tokens: int = 16384,

        # judge
        llm_judge: bool = False,

        seed: int = 42,
):
    # Path
    print("evaluate dataset: ", split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generation_file = os.path.join(output_dir, "generation.jsonl")
    result_file = os.path.join(output_dir, "result.log")
    tensor_parallel_size = 8

    # Load data
    if "Qwen2.5-Math" in base_model:
        max_model_len = 4096
        tensor_parallel_size = 4
        # if "72B" in base_model:
        #     tensor_parallel_size = 8
    if "deepseek" in base_model:
        max_model_len = 8192
    if "DeepSeek" in base_model:
        max_model_len = 8192
        tensor_parallel_size = 4
    if torch.cuda.device_count() < tensor_parallel_size:
        tensor_parallel_size = torch.cuda.device_count()
    if split == "math500":
        if data_file is not None and False:
            print("data_file:", data_file)
            test_dataset = read_jsonl(data_file)
        else:
            if data_dir is None:
                test_dataset = load_dataset(DATASET_ID, cache_dir="data/math_test")
            else:
                print(f"Loading data from {os.path.join(data_dir, DATASET_ID)}")
                test_dataset = load_from_disk(os.path.join(data_dir, DATASET_ID))
            test_dataset = test_dataset[split]
    else:
        with open(f"data/math_test/{split}/test.jsonl", "r") as fp:
            test_dataset = [json.loads(line) for line in list(fp)]

    for td in test_dataset:
        # print(td)
        if "problem" not in td and "question" in td:
            td["problem"] = td["question"]
        elif "problem" not in td and "Question" in td:
            td["problem"] = td["Question"]
        elif "problem" in td:
            pass
        else:
            raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    additional_prompt = ""
    if add_prompt is not None:
        additional_prompt = add_prompt
    if "deepseek" in base_model:
        additional_prompt = ""
    prefix_prompt = ""
    if add_prompt_prefix is not None:
        prefix_prompt = add_prompt_prefix
    prompts = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prefix_prompt + td["problem"] + additional_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for td in test_dataset
    ]
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    enable_lora = peft_path is not None
    print("prefix_prompt:\n", prefix_prompt)
    print("additional_prompt:\n", additional_prompt)
    print("Problem:\n", test_dataset[0]["problem"])
    print("Example\n", prompts[0])

    print("base_model:", base_model)
    print("tensor_parallel_size:", tensor_parallel_size)
    print("dtype:", dtype)
    print("seed:", seed)
    print("enable_lora:", enable_lora)
    print("max_model_len:", max_model_len)
    print("enforce_eager:", enforce_eager)
    print("num_scheduler_steps:", num_scheduler_steps)

    # load model
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        seed=seed,
        enable_lora=enable_lora,
        max_model_len=max_model_len,
        max_lora_rank=128,
        enforce_eager=enforce_eager,
        num_scheduler_steps=num_scheduler_steps,
        distributed_executor_backend="mp"
    )

    # generate
    print("maj_vote:", maj_vote)
    if maj_vote == 1:
        sampling_params = SamplingParams(temperature=0, max_tokens=max_generation_tokens)
    else:
        sampling_params = SamplingParams(n=maj_vote, temperature=0.7, max_tokens=max_generation_tokens, top_p=0.95)
    print("sampling_params:", sampling_params)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(
            lora_name="default",
            lora_int_id=1,
            lora_path=peft_path
        ) if peft_path is not None else None
    )
    assert len(outputs) == len(prompts)

    generations = []
    for td, o, p in zip(test_dataset, outputs, prompts):
        new_td = deepcopy(td)
        new_td["response"] = [r.text for r in o.outputs]
        new_td["prompt"] = p
        generations.append(new_td)

    # save generations
    write_jsonl(generation_file, generations)
    # save config
    config = {
        "llm_judge": llm_judge,
        "generation_file": generation_file,
        "result_file": result_file,
        "maj_vote": maj_vote,
        "bf16": bf16,
        "fp16": fp16,
        "seed": seed,
        "enforce_eager": enforce_eager,
        "num_scheduler_steps": num_scheduler_steps,
    }
    write_jsonl(os.path.join(output_dir, "config.json"), config)


def judge(config_file, args):
    # Load config
    print("config_file:", config_file)
    config = read_json(config_file)
    print("config:", config)
    llm_judge = config["llm_judge"]
    generation_file = config["generation_file"]
    result_file = config["result_file"]
    maj_vote = config["maj_vote"]
    bf16 = config["bf16"]
    fp16 = config["fp16"]
    seed = config["seed"]
    enforce_eager = config["enforce_eager"]
    num_scheduler_steps = config["num_scheduler_steps"]

    # Load generations
    generations = read_jsonl(generation_file)

    _judge(llm_judge, generations, generation_file, result_file, maj_vote, bf16, fp16, seed, enforce_eager,
           num_scheduler_steps, args=args)


if __name__ == "__main__":
    # fire.Fire()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file", type=str, default="config.json")
    argparser.add_argument("--type", type=str, default="eval")
    argparser.add_argument("--base_model", type=str, default="")
    argparser.add_argument("--chat_template_name", type=str, default="")
    argparser.add_argument("--output_dir", type=str, default="")
    argparser.add_argument("--bf16", type=bool, default=True)
    argparser.add_argument("--split", type=str, default="")
    argparser.add_argument("--llm_judge", type=bool, default=True)
    argparser.add_argument("--add_prompt", type=str, default="")
    argparser.add_argument("--judge_llm_path", type=str,
                           default="")

    args = argparser.parse_args()

    if args.type == "eval":
        eval(base_model=args.base_model, chat_template_name=args.chat_template_name,
             output_dir=args.output_dir, bf16=args.bf16, split=args.split,
             llm_judge=args.llm_judge, add_prompt=args.add_prompt
             )
    elif args.type == "judge":
        judge(args.config_file, args)
    else:
        raise NotImplementedError
