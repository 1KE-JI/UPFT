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


def get_urls(ips, ports):
    urls = ""
    for IP in ips:
        for port in ports:
            urls += f"http://{IP}:{port}/v1,"
    urls = urls[:-1]
    return urls
