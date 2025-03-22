from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
# from openrlhf.datasets.utils import zero_pad_sequences
from collections import defaultdict


def preprocess_data(data, input_template=None, input_key="input", output_key=None, apply_chat_template=None):
    if apply_chat_template:
        if output_key:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key] + data[output_key], tokenize=False)[len(prompt):]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt):]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
            self,
            dataset,
            tokenizer: Callable,
            max_length: int,
            strategy,
            input_template=None,
            pretrain_mode=False,
            num_processors=2,  # Specify the number of processors you want to use
            multiple_of=1,
            mode='train',
            args=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        cache_path = "-".join(args.data_path.split("/"))
        cache_path = f"new_cache/{cache_path}_ratio_{args.data_ratio}_{args.chat_template_name}_{args.planning_pruning}_planning_pruning_token_{args.planning_pruning_token}_prefix_length_{args.planning_prefix_tuning_length}_negative_mode_{args.negative_mode}_{args.max_len}.pt"
        self.strategy.print("cache_path:", cache_path)
        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        
        use_cache = False
        if len(dataset) > 20000:
            use_cache = True
        # use_cache = False
        if args.packing_samples:
            use_cache = False
        # use_cache = False
        self.strategy.print("use_cache:", use_cache)
        # self.dataset = dataset
        
        # processed_dataset = dataset.map(
        #     self.filter_data, remove_columns=, num_proc=num_processors
        # )
        # processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        # self.strategy.print("after filter data:", len(self.dataset))
        # self.dataset = processed_dataset
        # """
        if not os.path.exists(cache_path) or mode != 'train' or not use_cache:
            processed_dataset = dataset.map(
                self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
            )
            processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
            if mode == 'train' and use_cache:
                torch.save(processed_dataset, cache_path)
        else:
            self.strategy.print("load cache:", cache_path)
            processed_dataset = torch.load(cache_path)
        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.strategy.print("sft_type")
        if args.planning_pruning:
            self.sft_types = processed_dataset["sft_type"]
        # """
        self.args = args
        self.mode = mode
        self.print_debug = defaultdict(bool)
        self.print_flag = False

        # self.planning_prompt_ids_lens = processed_dataset["planning_prompt_ids_len"]

    def process_data(self, data):
        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {"prompt": prompt, "response": response, "prompt_ids_len": prompt_ids_len, "sft_type": data['sft_type']}
        # return {"prompt": prompt}

    def filter_data(self, data):
        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        # return {"prompt": prompt, "response": response, "prompt_ids_len": prompt_ids_len, "sft_type": data['sft_type']}
        return {"prompt": prompt}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        
        if True:
            prompt_ids_len = self.prompt_ids_lens[idx]
            prompt = self.prompts[idx]
            response = self.responses[idx]
            try:
                sft_type = self.sft_types[idx]
            except:
                sft_type = "supervised_tuning"
        else:
            data = self.dataset[idx]
            prompt, response = preprocess_data(
                data,
                None if self.pretrain_mode else self.input_template,
                self.input_key,
                self.output_key,
                apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            )
            
            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                    prompt = None
            else:
                prompt_ids_len = 0
            
            if self.args.planning_pruning:
                sft_type = data['sft_type']
            else:
                sft_type = "supervised_tuning"
        if not self.pretrain_mode:
            
            text = (prompt + response).rstrip("\n")
            flag = True
            if self.mode == 'train' and self.args.planning_pruning:
                flag = False
            if self.args.planning_pruning:
                if sft_type == "supervised_tuning":
                    flag = True
            if not flag:
                self.strategy.print()
                pruning_response = " ".join(response.split()[:self.args.planning_prefix_tuning_length])
                text = (prompt + pruning_response).rstrip("\n")
                if not self.args.without_ass_token:
                    text += " " + self.tokenizer.eos_token
            else:
                if not text.endswith(self.tokenizer.eos_token):
                    text += " " + self.tokenizer.eos_token

            if not self.print_debug[sft_type]:
                self.strategy.print(f"{sft_type} text:", text)
                self.print_debug[sft_type] = True
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if not flag and self.args.planning_pruning_token:
            input_token["input_ids"] = input_token["input_ids"][0][:prompt_ids_len+self.args.planning_prefix_tuning_length+1].unsqueeze(0)
            input_token["attention_mask"] = input_token["attention_mask"][0][:prompt_ids_len+self.args.planning_prefix_tuning_length+1].unsqueeze(0)
            if not self.print_flag:
                self.strategy.print("not flag and self.args.planning_pruning_token")
                self.strategy.print("input_token:", input_token)
                self.strategy.print("prompt_ids_len:", prompt_ids_len)
                self.print_flag = True
        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
        
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item()}
        if sft_type == "supervised_tuning":
            sft_type = torch.tensor(0)
        else:
            sft_type = torch.tensor(1)
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info, sft_type

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        sft_types = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info, sft_type in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            sft_types.append(sft_type)

        input_ids, pad_idx_begins = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id,
                                                       return_pad_idx_begin=True)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos, pad_idx_begins, sft_types

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:  # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos, [-1] * len(item_list)


def zero_pad_sequences(sequences, side: str = "left", value=0, return_pad_idx_begin=False):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    pad_idx_begins = []
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
        pad_idx_begins.append(seq.size(-1))
    if return_pad_idx_begin:
        return torch.stack(padded_sequences, dim=0), pad_idx_begins
    return torch.stack(padded_sequences, dim=0)
