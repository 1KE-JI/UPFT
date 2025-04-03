from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import os
import sys
import json
from tqdm import tqdm


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def write_txt(file: str, data_str: str):
    with open(file, "w") as f:
        f.write(data_str)


def read_jsonl(file):
    """
    Read a JSONL file.

    Args:
        file (str): The path to the JSONL file.

    Returns:
        List[dict]: A list of dictionaries, each representing a sample.
    """
    print("processing file:", file)
    if not os.path.exists(file):
        return []

    with open(file, "r") as f:
        return [json.loads(line) for line in tqdm(f)]


def append_jsonl(file: str, data: Union[dict, List[dict]]):
    """
    Append data to a JSONL file.

    Args:
        file (str): The path to the JSONL file.
        data (Union[dict, List[dict]]): The data to append.
    """
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass

    if isinstance(data, dict):
        data = [data]

    with open(file, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def write_jsonl(file: str, data: Union[dict, list[dict]]) -> None:
    """
    Write data to a JSONL file.

    Args:
        file (str): The path to the JSONL file.
        data (Union[dict, List[dict]]): The data to write.
    """
    if isinstance(data, dict):
        data = [data]

    with open(file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def get_data_path(dataset_name, data_ratio=-1):
    if dataset_name == "PRM12K":
        data_path = "data/Math/Llama-3.1-8B-Instruct_self_training_positive_gt.jsonl"
    elif dataset_name == "OMI2_600K":
        data_path = "data/Math/open_math_instruct_2.train.jsonl.dedup"
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