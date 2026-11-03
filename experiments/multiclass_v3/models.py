"""Train-only features and independent-cue bag learning for offline research."""
from collections import defaultdict
import copy
import numpy as np
import torch
from torch import nn
from multiclass_features import FeaturePreprocessor, MOTOR10
from multiclass_metrics import evaluate, cue_probabilities
from study_multiclass_transfer import person_class_cue_weights


def cue_groups(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['trial']].append(i)
    groups = [np.array(groups[k], dtype=int) for k in sorted(groups)]
    if any(len({rows[i]['label'] for i in g}) != 1 or
           len({rows[i]['subject'] for i in g}) != 1 for g in groups):
        raise ValueError('A cue must have one participant and one label.')
    return groups


def trial_weights(rows):
    groups = cue_groups(rows)
    representatives = [rows[g[0]] for g in groups]
    y = np.array([r['label'] for r in representatives])
    return groups, y, person_class_cue_weights(y, representatives)


class Features:
    def __init__(self, parameters):
        self.parameters = parameters

    @classmethod
    def fit(cls, cov, rows, channels, name, shrinkage=.3):
        chosen = channels if name.endswith('_all') else MOTOR10
        p = dict(name=name, indices=[channels.index(c) for c in chosen], channels=chosen)
        obj = cls(p)
        weights = person_class_cue_weights(np.array([r['label'] for r in rows]), rows)
        weights /= weights.sum()
        x = obj.subset(cov)
        if name.startswith('tangent'):
            p['tangent'] = FeaturePreprocessor.fit(x, weights, 'log_covariance', shrinkage).parameters
        else:
            f = obj.raw_power(x)
            p['mean'] = np.einsum('b,bf->f', weights, f)
            p['std'] = np.maximum(np.sqrt(np.einsum('b,bf->f', weights, (f-p['mean'])**2)), .1)
        return obj

    def subset(self, cov):
        i = self.parameters['indices']
        return np.asarray(cov)[:, :, i, :][:, :, :, i]

    def raw_power(self, cov):
        # Volts squared remain well above the numerical floor in accepted EEG.
        f = np.log(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-24))
        if self.parameters['name'].startswith('relative'):
            f -= f.mean(axis=-1, keepdims=True)
        return f.reshape(len(cov), -1)

    def transform(self, cov):
        p = self.parameters
        x = self.subset(cov)
        if 'tangent' in p:
            return FeaturePreprocessor(p['tangent']).transform(x)
        return ((self.raw_power(x)-p['mean'])/p['std']).astype(np.float32)


class CueNetwork(nn.Module):
    def __init__(self, inputs, classes, head, hidden=16, dropout=.5):
        super().__init__()
        self.layers = (nn.Linear(inputs, classes) if head == 'linear' else
                       nn.Sequential(nn.Dropout(dropout), nn.Linear(inputs, hidden),
                                     nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, classes)))

    def forward(self, x):
        return self.layers(x)


def predict(model, x):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.softmax(model(torch.from_numpy(x[a:a+256])), 1).numpy()
                               for a in range(0, len(x), 256)])


def diagnostics(p, rows, classes):
    y = np.array([r['label'] for r in rows])
    metrics = evaluate(p, y, rows, classes)
    truth, probs, _ = cue_probabilities(p, rows)
    metrics['cue_nll'] = float(-np.log(np.maximum(probs[np.arange(len(truth)), truth], 1e-12)).mean())
    scores = []
    for subject in sorted({r['subject'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['subject'] == subject]
        sub = evaluate(p[idx], y[idx], [rows[i] for i in idx], classes)
        scores.append(sub['cue_trials']['balanced_accuracy'])
    metrics['mean_person_cue_ba'] = float(np.mean(scores))
    return metrics


def selection_score(metrics):
    return (metrics['mean_person_cue_ba'], min(metrics['cue_trials']['recall'].values()),
            -metrics['cue_nll'])


def fit(x, rows, vx, val_rows, classes, head, config, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CueNetwork(x.shape[1], len(classes), head, config['hidden_units'], config['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    groups, truth, weights = trial_weights(rows)
    best = None
    history = []
    wait = 0
    for epoch in range(1, config['epochs']+1):
        model.train()
        order = rng.permutation(len(groups))
        epoch_loss = []
        for a in range(0, len(order), config['batch_cues']):
            selected = order[a:a+config['batch_cues']]
            indices, bag = [], []
            for b, j in enumerate(selected):
                g = groups[j]
                chosen = rng.choice(g, min(len(g), config['windows_per_cue']), replace=False)
                indices.extend(chosen)
                bag.extend([b]*len(chosen))
            bag = torch.tensor(bag)
            optimizer.zero_grad()
            wp = torch.softmax(model(torch.from_numpy(x[indices])), 1)
            sums = torch.zeros((len(selected), len(classes))).index_add(0, bag, wp)
            counts = torch.bincount(bag, minlength=len(selected)).float()[:, None]
            logp = (sums/counts).clamp_min(1e-12).log()
            nll = -logp[torch.arange(len(selected)), torch.from_numpy(truth[selected])]
            smooth = config['label_smoothing']
            nll = (1-smooth)*nll-smooth*logp.mean(1)
            w = torch.as_tensor(weights[selected], dtype=torch.float32)
            loss = (w*nll).sum()/w.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        vp = predict(model, vx)
        metrics = diagnostics(vp, val_rows, classes)
        row = dict(epoch=epoch, sampled_cue_loss=float(np.mean(epoch_loss)),
                   validation_ba=metrics['mean_person_cue_ba'], validation_nll=metrics['cue_nll'])
        if epoch == 1 or epoch % 10 == 0:
            tm = diagnostics(predict(model, x), rows, classes)
            row.update(train_ba=tm['mean_person_cue_ba'], train_nll=tm['cue_nll'])
        history.append(row)
        score = selection_score(metrics)
        if best is None or score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()), epoch)
            wait = 0
        else:
            wait += 1
        if wait >= config['patience']:
            break
    model.load_state_dict(best[1])
    return model, dict(seed=seed, selected_epoch=best[2], history=history,
                       train=diagnostics(predict(model, x), rows, classes),
                       validation=diagnostics(predict(model, vx), val_rows, classes))


def alignment_reference(cov, rows):
    """EA reference from explicitly supplied past data only; no labels required."""
    groups = cue_groups(rows)
    weights = np.zeros(len(rows))
    for group in groups:
        weights[group] = 1/(len(groups)*len(group))
    ref = np.einsum('b,bkij->ij', weights, np.asarray(cov, dtype=np.float64))
    # Normalize before the relative eigenvalue floor: raw EEG is ~1e-11 V².
    scale = np.trace(ref)/ref.shape[-1]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid alignment reference.')
    values, vectors = np.linalg.eigh((ref+ref.T)*.5/scale)
    values = np.maximum(values, .05)
    return ((vectors*values[None, :]**-.5)@vectors.T/np.sqrt(scale)).astype(np.float64)


def align(cov, reference):
    return (reference[None, None]@cov@reference.T[None, None]).astype(np.float32)


class Runner:
    def __init__(self, payload):
        self.features = Features(payload['features'])
        self.models = []
        for state in payload['states']:
            model = CueNetwork(payload['inputs'], len(payload['classes']), payload['head'],
                               payload['hidden_units'], payload['dropout'])
            model.load_state_dict(state)
            self.models.append(model)

    def predict(self, cov):
        x = self.features.transform(cov)
        return np.mean([predict(model, x) for model in self.models], axis=0)
