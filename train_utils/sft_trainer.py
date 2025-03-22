import os
from abc import ABC

import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm

from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
import random

from train_utils.loss import GPTLMLoss_Pro
from deepspeed.runtime.engine import DeepSpeedEngine

    

class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            eval_dataloader,
            in_domain_train_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            reduction=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.in_domain_train_dataloader = in_domain_train_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)
        if reduction is not None:
            self.loss_fn = GPTLMLoss_Pro(ring_attn_group=self.strategy.ring_attn_group, reduction=reduction)
        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None

        self.print_debug = False
        self.print_do_planning_tuning = False
        
        self.loss_sum = 10000000

        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
            # args.save_steps = num_update_steps_per_epoch  # Save once per epoch
            if self.epochs > 2:
                args.save_steps = num_update_steps_per_epoch * 3

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompt_id_lens, inputs, attention_masks, infos, pad_idx_begins, sft_types in self.train_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                # for label in labels:
                #     self.strategy.print("label:", label)
                # self.strategy.print("pad_idx_begin:", pad_idx_begin)
                if not self.print_debug:
                    self.strategy.print(f"****************** Step {step} ******************")
                    self.strategy.print("****************** before ******************")
                    self.strategy.print("eos_token_id:", self.tokenizer.eos_token_id)
                    self.strategy.print("labels[0]:", labels[0].cpu().tolist())
                    self.strategy.print("labels[-1]:", labels[-1].cpu().tolist())
                    if labels.shape[0] > 5:
                        self.strategy.print("labels[3]:", labels[3].cpu().tolist())
                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        # self.strategy.print("prompt_id_lens:", prompt_id_lens)
                        # self.strategy.print("infos['input_length']:", infos["input_length"])
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index: index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        if self.args.mpm_enable:
                            enable_flag = random.randint(1, 100)
                            if enable_flag <= int(100 * self.args.mpm_ratio):
                                # label[source_len + self.args.mpm_prefix_length:pad_idx_begin - 1]
                                # self.loss_fn.IGNORE_INDEX
                                labels = self._mask_tokens(self.tokenizer, labels, args=self.args,
                                                           source_len=prompt_id_lens,
                                                           pad_idx_begins=pad_idx_begins)
                        for label, source_len, pad_idx_begin, sft_type in zip(labels, prompt_id_lens, pad_idx_begins, sft_types):
                            if not args.compute_q_loss:
                                label[:source_len] = self.loss_fn.IGNORE_INDEX
                            # planning tuning
                            if self.args.planning_pruning or (
                                    self.args.planning_prefix_tuning_length == 0 and self.args.planning_suffix_tuning_length == 0):
                                if not self.print_do_planning_tuning:
                                    self.strategy.print(
                                        "self.args.planning_prefix_tuning_length == 0 and self.args.planning_suffix_tuning_length == 0")
                                    if self.args.planning_pruning:
                                        self.strategy.print("self.args.planning_pruning: True")
                                    self.print_do_planning_tuning = True
                            else:
                                # if self.args.planning_suffix_tuning_length == -1 or self.args.planning_suffix_tuning_length == 0:
                                #     label[source_len+self.args.planning_prefix_tuning_length:] = self.loss_fn.IGNORE_INDEX
                                # else:
                                if pad_idx_begin - self.args.planning_suffix_tuning_length > source_len + self.args.planning_prefix_tuning_length:
                                    # label[source_len+self.args.planning_prefix_tuning_length:pad_idx_begin-self.args.planning_suffix_tuning_length] = self.loss_fn.IGNORE_INDEX
                                    label[
                                    source_len + self.args.planning_prefix_tuning_length:pad_idx_begin - 1 - self.args.planning_suffix_tuning_length] = self.loss_fn.IGNORE_INDEX
                                    label[pad_idx_begin - 1] = self.loss_fn.IGNORE_INDEX
                            if "unigram" in self.args.planning_pruning_mode:
                                if sft_type == 1:
                                    # print("sft_type:", sft_type)
                                    position = int(self.args.planning_pruning_mode.split("_")[1])
                                    
                                    if "random" in self.args.planning_pruning_mode:
                                        position_begin = int(self.args.planning_pruning_mode.split("_")[-2])
                                        position_end = int(self.args.planning_pruning_mode.split("_")[-1])
                                        if position_end > position:
                                            raise ValueError("position_end must be shorter than position")
                                        span_length = position_end - position_begin+1
                                        random_position = random.randint(0, span_length-1)
                                        random_position = random_position + position_begin
                                        
                                        label[source_len: source_len+random_position-1] = self.loss_fn.IGNORE_INDEX
                                        label[source_len+random_position: pad_idx_begin-1] = self.loss_fn.IGNORE_INDEX
                                    else:
                                        label[source_len: source_len+position-1] = self.loss_fn.IGNORE_INDEX
                                        label[source_len+position:pad_idx_begin-1] = self.loss_fn.IGNORE_INDEX
                                else:
                                    # print("sft_type:", sft_type)
                                    pass

                if not self.print_debug:
                    try:
                        self.strategy.print(f"args.planning_pruning_mode: {self.args.planning_pruning_mode}")
                        self.strategy.print("****************** after ******************")
                        self.strategy.print("compute_q_loss:", args.compute_q_loss)
                        if "random" in self.args.planning_pruning_mode and "unigram" in self.args.planning_pruning_mode:
                            self.strategy.print("position_begin:", position_begin)
                            self.strategy.print("position_end:", position_end)
                            self.strategy.print("span_length:", span_length)
                            self.strategy.print("random_position:", random_position)
                        self.strategy.print("labels[0]:", labels[0].cpu().tolist())
                        self.strategy.print("labels[-1]:", labels[-1].cpu().tolist())
                        if labels.shape[0] > 5:
                            self.strategy.print("labels[3]:", labels[3].cpu().tolist())

                        if not self.args.debug:
                            self.print_debug = True
                    except:
                        pass

                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                loss_tensor = torch.tensor([loss.item()], device=torch.cuda.current_device())
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                if (torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any()) and self.strategy.is_rank_0:
                    print("Warning: Loss is NaN. Skipping this iteration.")
                    
                    print("****************** after ******************")
                    print(f"args.planning_pruning_mode: {self.args.planning_pruning_mode}")
                    print("compute_q_loss:", args.compute_q_loss)
                    print("inputs:", inputs)
                    if "random" in self.args.planning_pruning_mode and "unigram" in self.args.planning_pruning_mode:
                        print("position_begin:", position_begin)
                        print("position_end:", position_end)
                        print("span_length:", span_length)
                        print("random_position:", random_position)
                        print("pad_idx_begin:", pad_idx_begin)
                        print("source_len:", source_len)
                    print("labels[0]:", labels[0].cpu().tolist())
                    print("labels[-1]:", labels[-1].cpu().tolist())
                    self.model.zero_grad()
                    if isinstance(self.model, DeepSpeedEngine):
                        self.model.allreduce_gradients()
                    continue
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
                
                
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                loss_sum = self.evaluate(self.eval_dataloader, global_step, flag="dev")
                """
                if loss_sum < self.loss_sum:
                    self.loss_sum = loss_sum
                    # if self.strategy.is_rank_0():
                    self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, "best"))
                """
                    
            if len(self.in_domain_train_dataloader) > 0 and False:
                self.evaluate(self.in_domain_train_dataloader, global_step, flag="train_subset")
            
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0, flag="dev"):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompt_id_lens, inputs, attention_masks, infos, pad_idx_begins, _ in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index: index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {f"{flag} gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
        return loss_sum

    def get_loss_ppl(self, eval_dataloader, steps=0):
        times = 0
        self.debug_print_get_loss_ppl = False
        self.model.eval()
        
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            results = []
            for prompt_id_lens, inputs, attention_masks, infos, pad_idx_begins, _ in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index: index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss, shift_logits, shift_labels, logprobs, true_logprobs = self.loss_fn(output.logits, labels)
                
                ## 
                batch_size, seq_len = labels.size()
                total_loss_per_token = loss.view(batch_size, seq_len - 1)
                # Create a mask to ignore padding tokens (where labels == -100)
                mask = shift_labels.view(batch_size, seq_len - 1) != -100
                # Count the number of active elements (i.e., not-padded) per sample
                num_active_elements_per_sample = mask.sum(dim=1)
                # Avoid division by zero
                num_active_elements_per_sample[num_active_elements_per_sample == 0] = 1
                # Calculate the average loss per sample
                
                # per sample
                total_loss_per_sample = total_loss_per_token.masked_fill(~mask, 0).sum(dim=1)
                avg_loss_per_sample = total_loss_per_sample / num_active_elements_per_sample
                ppl_per_sample = torch.exp(avg_loss_per_sample)
                
                # per ppl
                total_loss_per_token = total_loss_per_token
                ppl_per_token = torch.exp(total_loss_per_token)
                
                if not self.debug_print_get_loss_ppl:
                    if self.strategy.is_rank_0():
                        print("loss.shape:", loss.shape)
                        print("shift_logits.shape:", shift_logits.shape)
                        print("shift_labels.shape:", shift_labels.shape)
                        print("logprobs.shape:", logprobs.shape)
                        
                        print("ppl_per_sample.shape:", ppl_per_sample.shape)
                        print("total_loss_per_sample.shape:", total_loss_per_sample.shape)
                        print("true_logprobs.shape:", true_logprobs.shape)
                        
                        print("total_loss_per_token.shape:", total_loss_per_token.shape)
                        print("ppl_per_token.shape:", ppl_per_token.shape)
                        self.debug_print_get_loss_ppl = True
                    
                        # loss.shape: torch.Size([250])
                        # shift_logits.shape: torch.Size([2, 125, 128256])
                        # shift_labels.shape: torch.Size([2, 125])
                        # logprobs.shape: torch.Size([2, 125, 128256])
                        
                        # ppl_per_sample.shape: torch.Size([2])
                        # total_loss_per_sample.shape: torch.Size([2])
                        # true_logprobs.shape: torch.Size([2, 125])
                        
                        # total_loss_per_token.shape: torch.Size([2, 125])
                        # ppl_per_token.shape: torch.Size([2, 125])
                result = []
                for i in range(batch_size):
                    _dict = {}
                    #prompt_id_lens, inputs, attention_masks, infos, pad_idx_begins
                    _dict["prompt_id_lens"] = prompt_id_lens[i]
                    _dict["inputs"] = inputs[i].tolist()
                    _dict["attention_masks"] = attention_masks[i].tolist()
                    # _dict["infos"] = infos[i]
                    _dict["pad_idx_begins"] = pad_idx_begins[i]
                    # metrics
                    _dict["ppl_per_sample"] = ppl_per_sample[i].cpu().tolist()
                    # _dict["logprobs"] = logprobs[i].cpu().tolist()
                    _dict["true_logprobs"] = true_logprobs[i].cpu().tolist()
                    _dict["total_loss_per_token"] = total_loss_per_token[i].cpu().tolist()
                    _dict["avg_loss_per_sample"] = avg_loss_per_sample[i].cpu().tolist()
                    result.append(_dict)
                    
                    
                results.extend(result)
                times += 1
                # loss_sum += loss.item()
                # bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                # logs = self.strategy.all_reduce(bar_dict)
                # step_bar.set_postfix(logs)

            # if self.strategy.is_rank_0():
            #     if self._wandb is not None:
            #         logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
            #         self._wandb.log(logs)
            #     elif self._tensorboard is not None:
            #         for k, v in logs.items():
            #             self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        all_features = [None] * dist.get_world_size()
        
        dist.all_gather_object(all_features, results)
        all_features = [item for sublist in all_features for item in sublist]
        return all_features    
        self.model.train()  # reset model state
        
    
    def _mask_tokens(self, tokenizer, labels, args=None, source_len=[], pad_idx_begins=[]):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        # mask_prob = args.mpm_p
        # mute_prob = int(args.mpm_mode[0]) / 10
        # replace_prob = int(args.mpm_mode[1]) / 10

        #########################################################################################################
        # We sample a few tokens in each sequence for masked-LM reasoning training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, args.mpm_p)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        for idx, label, pad_idx_begin in zip(list(range(len(labels))), labels, pad_idx_begins):
            for i in range(pad_idx_begin, len(label)):
                masked_indices[idx][i] = 1
            masked_indices[idx][pad_idx_begin - 1] = 1
        for idx in range(len(masked_indices)):
            for i in range(0, source_len[idx] + args.mpm_prefix_length):
                masked_indices[idx][i] = 1
        if not self.print_debug:
            self.strategy.print("****************** masked_indices ******************")
            self.strategy.print("masked_indices[0]:", masked_indices[0].cpu().tolist())
        ignore_value = -100
        labels[~masked_indices] = ignore_value
        return labels
