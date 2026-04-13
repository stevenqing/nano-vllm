import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Fast path: all greedy
        if (temperatures < 1e-10).all():
            return logits.argmax(dim=-1)
        logits = logits.float()
        greedy_mask = (temperatures < 1e-10)
        safe_temps = temperatures.clamp(min=1e-10)
        logits = logits.div_(safe_temps.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        gumbel_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        sample_tokens = torch.where(greedy_mask, greedy_tokens, gumbel_tokens)
        return sample_tokens
