import argparse
import json
import math
import os
from datetime import datetime
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from train_utils.sft_trainer import SFTTrainer
from inference.chat_template import CHAT_TEMPLATE

from datasets import Dataset
from train_utils.sft_dataset import SFTDataset
from transformers import set_seed
from inference.eval import read_json, read_jsonl


def evenly_select_and_keep_rest(data, num_to_select):
    n = len(data)
    if num_to_select >= n:
        return data, []
    
    step = (n - 1) / (num_to_select - 1) if num_to_select != 1 else 0
    selected_data = []
    unselected_data = []
    
    for i in range(n):
        if i % step < 1 or abs(i - round(i / step) * step) < 1e-5:
            selected_data.append(data[i])
        else:
            unselected_data.append(data[i])
            
    return selected_data, unselected_data


def find_max_global_step_directory(directory):
    max_step = -1
    max_directory = None
    for entry in os.listdir(directory):
        if entry.startswith("global_step"):
            try:
                step = int(entry.split("step")[-1])
                if step > max_step:
                    max_step = step
                    max_directory = entry
            except ValueError:
                continue
    return max_directory


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    set_seed(args.seed)
    use_custom_ds_config = False
    if args.zero_stage == 3 and use_custom_ds_config:
        ds_config = read_json(os.path.join("ds_config", "zero3_offload.json"))
    else:
        ds_config = strategy.get_ds_train_config(is_actor=True)
    strategy.print("ds_config: ", ds_config)
    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=ds_config,
        packing_samples=args.packing_samples,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    tokenizer.chat_template = CHAT_TEMPLATE[args.chat_template_name]
    # Set pad_token if needed
    pad_token_setting = "eos"
    if (tokenizer.pad_token_id is None):
        if pad_token_setting == "eos":
            assert tokenizer.eos_token_id is not None
            strategy.print("Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif pad_token_setting == "unk":
            assert tokenizer.unk_token_id is not None
            strategy.print("Setting pad_token to unk_token.")
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif pad_token_setting == "custom":
            pad_token_id = model2padid(base_model)
            strategy.print(
                f"Setting pad_token to {pad_token_id} ({tokenizer.decode(pad_token_id)})"
            )
            tokenizer.pad_token = tokenizer.decode(pad_token_id)
            tokenizer.pad_token_id = pad_token_id
        else:
            raise ValueError(f"Invalid pad_token_setting: {pad_token_setting}")
    strategy.print(model)
    strategy.print("data_path:", args.data_path)
    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    tasks = args.tasks.split("_")
    # prepare for data and dataset
    load_from_hf = False
    if not args.planning_pruning:
        args.add_prompt = ""
    if args.add_prompt == "1" or args.add_prompt == 1:
        args.add_prompt = ""
    # default
    if load_from_hf:
        train_data, eval_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            train_split=args.train_split,
            eval_split=args.eval_split,
        )
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    # custom
    else:
        raw_datasets = {
            "train": [],
            "test": [],
            "in_domain_train": []
        }
        planning_tuning_count = 0
        supervised_tuning_count = 0
        strategy.print(f"tasks：{tasks}")

        # unsupervised setting
        if "noposterior" in tasks:
            strategy.print("loading noposterior data")
        
            data_path = args.data_path
            strategy.print(data_path)
            try:
                with open(data_path, encoding='utf-8') as fp:
                    sft_dataset = json.load(fp)
            except:
                with open(data_path, encoding='utf-8') as fp:
                    sft_dataset = [json.loads(line) for line in tqdm(list(fp))]
        
            threshold = int(args.planning_pruning_ratio*len(sft_dataset))
            
            strategy.print("threshold:", threshold)
            strategy.print(sft_dataset[0])
            
            for pruning_flag, data in tqdm(enumerate(sft_dataset)):
                if "query" in data:
                    data['question'] = data['query']
                if "problem" in data and "question" not in data:
                    data['question'] = data['problem']
                if "noposterior" not in data and "solution" in data:
                    if isinstance(data['solution'], str):
                        data['noposterior'] = data['solution']
                    else:  
                        data['noposterior'] = data['solution'][0]
                if "response" in data:
                    if isinstance(data['response'], str):
                        data['noposterior'] = data['response']
                    else:
                        data['noposterior'] = data['response'][0]
                
                if "noposterior" not in data:
                    continue
                
                if len(data['question'].split())+len(data['noposterior'].split()) > args.max_len:
                    continue
                if pruning_flag < threshold and args.planning_pruning:
                    data['question'] = [{"role": "user", "content": data['question']+args.add_planning_prompt}]
                    data['sft_type'] = "planning_tuning"
                    data['response'] = [{"role": "assistant", "content": data['noposterior']}]
                    
                    planning_tuning_count += 1
                else:
                    data['question'] = [{"role": "user", "content": data['question']+args.add_prompt}]
                    data['sft_type'] = "supervised_tuning"
                    data['response'] = [{"role": "assistant", "content": data['noposterior']}]
                    supervised_tuning_count += 1
            strategy.print("noposterior")

            strategy.print("before check:", len(sft_dataset))

            if "check" in args.dataset_name:
                check_data = True
            check_data = False
            if check_data:
                def contain_chinese_char(s: str) -> bool:
                    return any(u'\u4e00' <= char <= u'\u9fff' for char in s)

                def contain_boxed(s: str) -> bool:
                    return "boxed" in s
                new_sft_dataset = []
                planning_tuning_count = 0
                supervised_tuning_count = 0
                for data in sft_dataset:
                    if "sft_type" not in data:
                        continue
                    if data["sft_type"] == "planning_tuning":
                        if not contain_chinese_char(data['noposterior']):
                            new_sft_dataset.append(data)
                            planning_tuning_count += 1
                    elif data["sft_type"] == "supervised_tuning":
                        if not contain_chinese_char(data['noposterior']) and contain_boxed(data['noposterior']):
                            new_sft_dataset.append(data)
                            supervised_tuning_count += 1
                    else:
                        raise ValueError("sft_type error")
                strategy.print("after clean, new_sft_dataset length:", len(new_sft_dataset))
                sft_dataset = new_sft_dataset

        # supervised setting
        if "sft" in tasks:
            strategy.print("loading positive data")
            data_path = args.data_path
            with open(data_path, encoding='utf-8') as fp:
                sft_dataset = [json.loads(line) for line in list(fp)]
            if "OpenMathInstruct2_self_training" in data_path:
                print("before", len(sft_dataset))
                new_sft_dataset = []
                for i in sft_dataset:
                    if i.get("matched", False) and "\\boxed" in i['response']:
                        new_sft_dataset.append(i)
                sft_dataset = new_sft_dataset
                print("after", len(sft_dataset))

            for data in sft_dataset:
                if "nvidia" in data_path:
                    data['response'] = data['generated_solution']
                elif "open_math_instruct_2" in data_path:
                    data['response'] = data['answer']
            threshold = int(args.planning_pruning_ratio*len(sft_dataset))
            for pruning_flag, data in enumerate(sft_dataset):
                if "query" in data:
                    data['question'] = data['query']
                if "response" not in data and "solution" in data:
                    if isinstance(data['solution'], list):
                        data['response'] = data['solution'][0]
                    else:
                        data['response'] = data['solution']
                if pruning_flag < threshold and args.planning_pruning:
                    data['question'] = [{"role": "user", "content": data['question']+args.add_planning_prompt}]
                    data['sft_type'] = "planning_tuning"
                    planning_tuning_count += 1
                else:
                    data['question'] = [{"role": "user", "content": data['question']+args.add_prompt}]
                    data['sft_type'] = "supervised_tuning"
                    supervised_tuning_count += 1
                data['response'] = [{"role": "assistant", "content": data['response']}]

        if args.data_ratio == -1:
            sft_dataset = sft_dataset
        else:
            sft_dataset = sft_dataset[:int(args.data_ratio*len(sft_dataset))]
        if len(sft_dataset) > 20000:
            dev_length = 2000
            in_domain_length = 4000
        else:
            dev_length = int(len(sft_dataset) * 0.1)
            in_domain_length = 500
        selected_data, unselected_data = evenly_select_and_keep_rest(sft_dataset, dev_length)
        if len(sft_dataset) > 20000:
            train_data = unselected_data
        else:
            train_data = sft_dataset
        eval_data = selected_data
        in_domain_train_data, _ = evenly_select_and_keep_rest(train_data, in_domain_length)
        
        strategy.print("train_data:", len(train_data))
        strategy.print("eval_data:", len(eval_data))
        strategy.print("in_domain_train_data:", len(in_domain_train_data))
        
        raw_datasets['train'].extend(train_data)
        raw_datasets['test'].extend(eval_data)
        raw_datasets['in_domain_train'].extend(in_domain_train_data)
        
        strategy.print("threshold:", threshold)
        strategy.print("planning_tuning_count:", planning_tuning_count)
        strategy.print("supervised_tuning_count:", supervised_tuning_count)
        
        train_data = Dataset.from_list(raw_datasets["train"])
        eval_data = Dataset.from_list(raw_datasets["test"])
        in_domain_train_data = Dataset.from_list(raw_datasets["in_domain_train"])
        
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        mode='train',
        args=args,
    )
    strategy.print("Length of train data:", len(train_data))
    strategy.print("Length of eval data:", len(eval_data))

    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        mode='eval',
        args=args,
    )
    
    in_domain_train_dataset = SFTDataset(
        in_domain_train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        mode='eval',
        args=args,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )
    in_domain_train_dataloader = strategy.setup_dataloader(
        in_domain_train_dataset,
        args.micro_train_batch_size,
        True,
        False,
        in_domain_train_dataset.packing_collate_fn if args.packing_samples else in_domain_train_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    args.ckpt_path = args.save_path
    strategy.print("args.ckpt_path:", args.ckpt_path)
    
    tag = find_max_global_step_directory(args.ckpt_path)

    strategy.print("tag:", tag)
    if args.load_checkpoint and os.path.exists(args.ckpt_path) and tag is not None:
        _, states = strategy.load_ckpt(model.model, args.ckpt_path, tag=tag)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
        cur_save_path = os.path.join(args.save_path, "ckpt_{}".format(tag))
        strategy.print("cur_save_path:", cur_save_path)
        strategy.save_model(model, tokenizer, cur_save_path)

    if args.save_only:
        exit(0)

    strategy.print("save_path:", args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        in_domain_train_dataloader=in_domain_train_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=100)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default="")
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # My SFT parameters
    parser.add_argument("--data_ratio", type=float, default=-1)
    parser.add_argument("--planning_pruning_ratio", type=float, default=-1)

    # PlanTuning parameters
    parser.add_argument("--chat_template_name", type=str, default="llama-3.1-chat")
    parser.add_argument("--tasks", type=str, default="dqa_qa")
    parser.add_argument("--data_path", type=str, default="")

    parser.add_argument("--planning_prefix_tuning_length", type=int, default=32)
    parser.add_argument("--planning_suffix_tuning_length", type=int, default=0)
    parser.add_argument("--add_prompt", type=str, default="")
    parser.add_argument("--add_planning_prompt", type=str, default="")
    parser.add_argument("--planning_pruning", type=int, default=0)
    parser.add_argument("--negative_mode", type=str, default="negative_planning", choices=["negative_none", "negative_planning", "negative_whole", "noposterior"])
    parser.add_argument("--dataset_name", type=str, default="")
    
    parser.add_argument("--planning_pruning_token", type=int, default=0)
    parser.add_argument("--planning_pruning_mode", type=str, default="unigram_4_random_1_4")
    parser.add_argument("--compute_q_loss", type=int, default=0)
    
    parser.add_argument("--pruning_ratio", type=int, default=0)
    parser.add_argument("--without_ass_token", type=int, default=0)
    # mpm
    parser.add_argument("--mpm_enable", type=int, default=0)
    parser.add_argument("--mpm_p", type=float, default=-1)
    parser.add_argument("--mpm_mode", type=str, default="811")
    parser.add_argument("--mpm_ratio", type=float, default=-1)
    parser.add_argument("--mpm_prefix_length", type=int, default=8)

    # debug
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save_only", type=int, default=1)
    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    # TODO: [packing samples]
    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    args.wandb_project = args.wandb_project + f"_UPFT{args.planning_pruning}_ngram_{args.dataset_length}"
    args.wandb_run_name = args.wandb_run_name + f"_{args.planning_pruning_mode}" + f"_{args.dataset_length}"
    args.eval_steps = args.save_steps
    train(args)
