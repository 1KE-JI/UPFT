import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, GenerationConfig
from transformers import set_seed

from vllm import LLM, SamplingParams
from tqdm import tqdm

from transformers.utils import logging
from typing import Union, Any
import random
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from chat_template import CHAT_TEMPLATE
from utils.utils import *
from utils.openmathinst_utils import extract_answer, math_equal



def evenly_select_and_keep_rest(data, num_to_select):
    """
    从给定的数据中等距离选择指定数量的数据点，并保留未选中的数据。
    
    :param data: 输入的一维数组或列表
    :param num_to_select: 需要选择的数据点数量
    :return: (选中的数据点组成的列表, 未选中的数据点组成的列表)
    """
    n = len(data)
    if num_to_select >= n:
        return data, []  # 如果选取数量大于等于数据量，则全部选中
    
    step = (n - 1) / (num_to_select - 1) if num_to_select != 1 else 0
    selected_data = []
    unselected_data = []
    
    for i in range(n):
        if i % step < 1 or abs(i - round(i / step) * step) < 1e-5:  # 精确判断是否应该选取当前索引
            selected_data.append(data[i])
        else:
            unselected_data.append(data[i])
            
    return selected_data, unselected_data


def get_data_path(dataset_name, data_ratio=-1):
    if dataset_name == "1W":
        data_path = "data/Math/Llama-3.1-8B-Instruct_self_training_positive_gt.jsonl"
        data_ratio = -1
    elif dataset_name == "1W_noposterior":
        data_path = "rollout_outputs/noposterior_sync_-1_rollout_dataset_llama1W_prefix_length_0_n_32_data_length_11484.json"
        data_ratio = -1
    elif dataset_name == "Qwen25_1W":
        data_path = "data/Math/Qwen2.5-Math-7B-Instruct_one_shot_step_baseline.jsonl"
        data_ratio = -1
    elif dataset_name == "Qwen25_1W_noposterior":
        data_path = "rollout_outputs/sync_-1_rollout_dataset_Qwen25_1W_prefix_length_0_n_1_data_length_11405.json"
        data_ratio = -1
    elif dataset_name == "60W":
        data_path = "data/Math/open_math_instruct_2.train.jsonl.dedup"
        data_ratio = -1
    elif dataset_name == "60W_self_training":
        data_path = "data/sampling/OpenMathInstruct2_self_training_v2.jsonl"
        data_ratio = -1
    elif dataset_name == "100W":
        data_path = "/root/workspace/hf_datasets/nvidia/OpenMathInstruct-1/correct_solutions/train.jsonl"
        data_ratio = -1
    elif dataset_name == "hard":
        data_path = "data/Math/train.annotated_subset_356748.jsonl"
        data_ratio = -1
    elif dataset_name == "limo":
        data_path = "data/Math/limo.jsonl"
        data_ratio = -1
    elif dataset_name == "math500":
        data_path = "data/math_test/math500/test.jsonl"
        data_ratio = -1
    elif dataset_name == "s1":
        data_path = "data/s1/gen_s1_59k.jsonl"
        data_ratio = -1
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))
    return data_path, data_ratio


def contain_chinese_char(s: str) -> bool:
    return any(u'\u4e00' <= char <= u'\u9fff' for char in s)
    
    
def contain_boxed(s: str) -> bool:
    return "boxed" in s


def clean_data():
    input_file_path = "rollout_result/tgt_Qwen2.5-7B-Instruct_1W_data_length_11484_max16384_n_1.json"
    # input_file_path = "rollout_result/tgt_Llama-3.2-3B-Instruct_1W_data_length_11484_max16384_n_1.json"
    # input_file_path = "rollout_result/tgt_Llama-3.3-70B-Instruct_1W_data_length_11484_max16384_n_1.json"
    tgt_file_path = input_file_path.replace(".json", "_clean.json")
    print("input_file_path:", input_file_path)
    
    datasets = read_jsonl(input_file_path)
    
    print("len(datasets):", len(datasets))
    print("datasets[0].keys():", datasets[0].keys())

    new_sft_dataset = []
    subset_whole_solution_datasets = []
    subset_planning_solution_datasets = []
    
    planning_tuning_count = 0
    supervised_tuning_count = 0
    
    max_length = 0
    
    question_length_list = []
    response_length_list = []
    
    for data in tqdm(datasets):
        if isinstance(data['response'], list):
            data['response'] = data['response'][0]
            
        if contain_chinese_char(data['response']):
            continue
    
        if not contain_boxed(data['response']):
            subset_planning_solution_datasets.append(data)
            planning_tuning_count += 1
        else:
            subset_whole_solution_datasets.append(data)
            supervised_tuning_count += 1
        question_length_list.append(len(data['query'].split()))
        response_length_list.append(len(data['response'].split()))
        
    new_sft_dataset.extend(subset_planning_solution_datasets)
    new_sft_dataset.extend(subset_whole_solution_datasets)

    print(new_sft_dataset[0].keys())
    
    print("planning_tuning_count:", planning_tuning_count)
    print("supervised_tuning_count:", supervised_tuning_count)
    print("new_sft_dataset length:", len(new_sft_dataset))
    
    print("Avg. question length:", sum(question_length_list)/len(question_length_list))
    print("Avg. response length:", sum(response_length_list)/len(response_length_list))
    print("Max. question length:", max(question_length_list))
    print("Max. response length:", max(response_length_list))
    
    print("tgt_file_path:", tgt_file_path)
    write_jsonl(tgt_file_path, new_sft_dataset)
    
    
if __name__ == "__main__":
    clean_data()
