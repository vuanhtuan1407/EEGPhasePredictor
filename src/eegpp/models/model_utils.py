import torch


def preprocess_inp(x, emb_dim, turnoff=None):
    if turnoff is not None:
        for turnoff_idx in turnoff:
            turnoff_size = x[turnoff_idx].shape
            x[turnoff_idx] = torch.zeros(turnoff_size, dtype=x.dtype, device=x.device)
    x = x.unsqueeze(-1)
    x = x.repeat(1, 1, 1, emb_dim)
    return x
