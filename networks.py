"""EEGNet architecture compatible with the supplied fc checkpoints."""
import torch
from torch import nn

class EEGNet(nn.Module):
    """
    Compact Convolutional Neural Network for EEG decoding.
    Reference: Lawhern et al., 2018.
    """
    def __init__(self, n_ch: int = 20, n_t: int = 1024, n_cls: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kern: int = 128, drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop)
        )
        flat_size = self._get_flat_dim(n_ch, n_t)
        self.fc = nn.Linear(flat_size, n_cls)

    def _get_flat_dim(self, c: int, t: int) -> int:
        self.block1.eval()
        self.block2.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, c, t)
            out = self.block2(self.block1(dummy))
            return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))
