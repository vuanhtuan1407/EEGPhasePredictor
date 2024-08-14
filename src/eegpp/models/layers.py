import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(self, emb_dim: int, turnoff=None):
        """Preprocess input before feeding into models.

        Args:
            emb_dim:
            turnoff: Options[list[int]]: index of channel turned off. Indices must be in [0, 1, 2] relevant to [EEG, EMG, MOT]

        """
        super().__init__()
        self.emb_dim = emb_dim
        self.turnoff = turnoff

    def forward(self, x):
        # x = (3, batch, seq)
        if self.turnoff is not None:
            for turnoff_idx in self.turnoff:
                turnoff_size = x[turnoff_idx].shape
                x[turnoff_idx] = torch.zeros(turnoff_size, dtype=x.dtype, device=x.device)
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.emb_dim)
        return x


if __name__ == "__main__":
    ie = InputEmbedding(2, turnoff=[0])
    t = torch.randint(1, 5, (3, 2, 4))
    print(t.shape)
    print(ie(t).shape)
    print(t)
    print(ie(t))
