import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F



class GPTLMLoss_Pro(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None, reduction=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX, reduction=reduction)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Generate logprobs for labels
        logprobs = F.log_softmax(shift_logits, dim=-1)
        
        unsq_shift_labels = shift_labels.unsqueeze(-1)
        
        batch_size, seq_len, vocab_size = logprobs.shape
        flat_logprobs = logprobs.view(batch_size * seq_len, vocab_size)
        flat_labels = unsq_shift_labels.view(batch_size * seq_len, 1).squeeze(-1)
        true_logprobs = flat_logprobs[torch.arange(batch_size * seq_len), flat_labels].view(batch_size, seq_len)
        # print("shift_labels.shape:", shift_labels.shape)
        # print("unsq_shift_labels.shape:", unsq_shift_labels.shape)
        # print("logprobs.shape:", logprobs.shape)
        # print("true_logprobs.shape:", true_logprobs.shape)
        # exit(0)
        # true_logprobs = logprobs.gather(dim=-1, index=unsq_shift_labels).squeeze(-1)
        return loss, shift_logits, shift_labels, logprobs, true_logprobs
