"""Cue-balanced training-only PCA and exact inference-time layer folding."""
import copy
from collections import defaultdict, Counter
import numpy as np
import torch
from torch import nn


def fit_projection(features, records, components):
    """Fit PCA to independent training-cue means, not overlapping-window copies.

    Each class receives equal total weight and each cue equal weight within its
    class. Components are orthonormal; there is no whitening or eigenvalue gain.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(records) or not np.isfinite(x).all():
        raise ValueError('Expected finite feature rows matching cue provenance.')
    if not isinstance(components, (int, np.integer)) or components < 1:
        raise ValueError('PCA component count must be a positive integer.')
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[str(row['trial'])].append(i)
    if len(groups) < 2:
        raise ValueError('At least two independent training cues are required.')
    means, labels = [], []
    for trial in sorted(groups):
        indices = groups[trial]
        label = {int(records[i]['label']) for i in indices}
        if len(label) != 1:
            raise ValueError('One cue has conflicting class labels.')
        labels.append(label.pop())
        means.append(x[indices].mean(axis=0))
    means = np.stack(means)
    counts = Counter(labels)
    weights = np.array([1/(len(counts)*counts[label]) for label in labels])
    center = np.einsum('b,bf->f', weights, means)
    weighted = (means-center)*np.sqrt(weights[:, None])
    _, singular, vectors = np.linalg.svd(weighted, full_matrices=False)
    magnitude = max(float(singular[0]), float(np.max(abs(means))))
    tolerance = max(weighted.shape)*np.finfo(float).eps*magnitude
    rank = int(np.sum(singular > tolerance))
    effective = min(int(components), len(means)-1, rank)
    if effective < 1:
        raise ValueError('No between-cue variation is available for PCA.')
    return dict(center=center.astype(np.float32),
                components=vectors[:effective].astype(np.float32),
                requested_components=int(components), effective_components=effective,
                numerical_rank=rank, n_training_cues=len(means), whitening=False,
                fit_unit='class-balanced means of complete training cue trials')


def project(features, projection):
    x = np.asarray(features, dtype=np.float32)
    if projection is None:
        return x
    center = np.asarray(projection['center'], dtype=np.float32)
    components = np.asarray(projection['components'], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(center) or components.shape[1] != len(center):
        raise ValueError('PCA feature geometry differs from the fitted projection.')
    return ((x-center)@components.T).astype(np.float32)


class ProjectedHead(nn.Module):
    def __init__(self, inputdim, hidden, classes, dropout):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(inputdim, hidden),
                                  nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, x):
        return self.head(x)


def makehead(inputdim, hidden, classes, dropout):
    return ProjectedHead(inputdim, hidden, classes, dropout)


def fold_state_dict(state, projection):
    """Fold z=(x-center)P.T into the first affine layer without mutating state.

    Exact equivalence is for inference mode, where both dropout layers are off.
    The resulting state loads into the unchanged FeatureNetwork/ensemble runner.
    """
    result = copy.deepcopy(state)
    if projection is None:
        return result
    weight = state['head.1.weight']
    components = torch.as_tensor(projection['components'], dtype=weight.dtype, device=weight.device)
    center = torch.as_tensor(projection['center'], dtype=weight.dtype, device=weight.device)
    if weight.shape[1] != components.shape[0] or components.shape[1] != len(center):
        raise ValueError('Neural head and PCA projection dimensions do not match.')
    folded = weight@components
    result['head.1.weight'] = folded
    result['head.1.bias'] = state['head.1.bias']-folded@center
    return result
