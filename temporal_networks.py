"""Temporal convolutional networks for multiclass EEG decoding."""
import torch
from torch import nn


class TemporalEEGNet(nn.Module):
    """Learn spatial filters and changes in band power within a past EEG window."""

    def __init__(self, n_channels, n_classes, window_standardization=False):
        super().__init__()
        self.window_standardization = window_standardization
        self.temporal = nn.ModuleList(
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (17, 33, 65)
        )
        self.spatial = nn.Sequential(
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 24, (n_channels, 1), groups=12, bias=False),
            nn.BatchNorm2d(24),
        )
        self.power = nn.AvgPool1d(16)
        self.context = nn.Sequential(
            nn.Conv1d(24, 24, 5, padding=2, bias=False),
            nn.BatchNorm1d(24), nn.ELU(),
            nn.Conv1d(24, 24, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(24), nn.ELU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(2), nn.Flatten(), nn.Dropout(.4), nn.Linear(48, n_classes))

    def forward(self, x):
        if self.window_standardization:
            x = x - x.mean(dim=1, keepdim=True)
            channel_rms = x.square().mean(dim=2).sqrt()
            scale = channel_rms.median(dim=1).values.clamp_min(1e-6)
            x = x / scale[:, None, None]
        x = x.unsqueeze(1)
        filtered = self.spatial(torch.cat([branch(x) for branch in self.temporal], dim=1)).squeeze(2)
        power = torch.log1p(self.power(filtered.square()))
        return self.head(power + self.context(power))


class TemporalEEGEnsemble(nn.ModuleList):
    def __init__(self, n_channels, n_classes, member_states, window_standardization=False):
        super().__init__(TemporalEEGNet(n_channels, n_classes, window_standardization) for _ in member_states)
        for model, state in zip(self, member_states):
            model.load_state_dict(state)

    def forward(self, x):
        probabilities = torch.stack([model(x).softmax(dim=1) for model in self])
        return probabilities.mean(dim=0).clamp_min(1e-12).log()
