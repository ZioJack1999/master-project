"""Shared multiclass cue decisions and calibration-policy diagnostics."""
from collections import defaultdict
import numpy as np
from multiclass_decoder import MulticlassPolicy

FIXED_POLICY = dict(ema_alpha=1., dwell_updates=1, min_confidence=.5, min_margin=.05)


def cue_decision(packets, eligible_updates=None, n_classes=3):
    """Majority accepted decision, with rejection counted separately from Rest.

    At least four accepted updates and at least half of all eligible updates
    are required. An exact class-vote tie rejects the cue. The optional eligible
    count can retain low-quality updates that were omitted from ``packets``.
    """
    packets = list(packets)
    eligible = len(packets) if eligible_updates is None else eligible_updates
    if (isinstance(eligible, bool) or not isinstance(eligible, (int, np.integer))
            or eligible < len(packets)):
        raise ValueError('Eligible update count must include every supplied packet.')
    votes = []
    for packet in packets:
        value = packet.get('class_id', -1)
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or not -1 <= value < n_classes):
            raise ValueError(f'Decisions must be integer IDs between -1 and {n_classes-1}.')
        if packet.get('quality', 'ok') == 'ok' and value >= 0:
            votes.append(int(value))
    decision = -1
    if len(votes) >= 4 and 2*len(votes) >= eligible:
        counts = np.bincount(votes, minlength=n_classes)
        if np.sum(counts == counts.max()) == 1:
            decision = int(counts.argmax())
    return dict(class_id=decision, eligible_updates=int(eligible), accepted_updates=len(votes))


def mean_probability_decision(packets, n_classes=3):
    """Comparison rule that pools clean windows before choosing a class."""
    probabilities = [p['probabilities_raw'] for p in packets
                     if p.get('quality') == 'ok' and 'probabilities_raw' in p]
    if len(probabilities) < 4 or 2*len(probabilities) < len(packets):
        return -1
    scores = np.mean(probabilities, axis=0)
    if scores.shape != (n_classes,) or not np.isclose(scores.sum(), 1.) or np.sum(scores == scores.max()) != 1:
        return -1
    return int(scores.argmax())


def diagnostic_from_probabilities(probabilities, rows, classes, policy=None):
    """Apply the fixed deployed policy to ordered, clean calibration windows.

    Policy state restarts per extracted cue. With the fixed alpha1/dwell1 policy
    this is identical at each update; unobserved continuous boundary/artifact
    windows cannot be reconstructed and no prospective accuracy is implied.
    """
    p = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    n_classes = len(classes)
    if n_classes < 3 or len(set(classes)) != n_classes:
        raise ValueError('Declare at least three unique classes.')
    if (p.shape != (len(rows), n_classes) or not len(rows) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
        raise ValueError('Expected one valid class probability vector per calibration window.')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if 'trial' not in row or 'label' not in row:
            raise ValueError('Calibration windows require trial and label provenance.')
        groups[str(row['trial'])].append(i)
    truth, decisions, details = [], [], []
    resolved_policy = dict(FIXED_POLICY, **(policy or {}))
    for trial in sorted(groups):
        indices = groups[trial]
        labels = {rows[i]['label'] for i in indices}
        if len(labels) != 1:
            raise ValueError('One calibration cue has conflicting labels.')
        label = labels.pop()
        if isinstance(label, str):
            if label not in classes:
                raise ValueError('Unknown calibration class label.')
            label = classes.index(label)
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or not 0 <= label < n_classes:
            raise ValueError('Unknown calibration class label.')
        # Existing extracted records carry sample indexes. Timestamp ordering
        # supports other explicit calibration records; input order breaks ties.
        def order(i):
            value = rows[i].get('start', rows[i].get('timestamp', i))
            if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError('Calibration window order must be finite.')
            return float(value), i
        indices = sorted(indices, key=order)
        classifier = MulticlassPolicy(n_classes, resolved_policy)
        packets = []
        for i in indices:
            quality = rows[i].get('quality', 'ok')
            packets.append(dict(classifier.update(p[i], quality), quality=quality))
        result = cue_decision(packets, n_classes=n_classes)
        truth.append(int(label))
        decisions.append(result['class_id'])
        details.append(dict(trial=trial, true_class=classes[label],
            decision=classes[result['class_id']] if result['class_id'] >= 0 else 'rejected', **result))
    y, prediction = np.array(truth), np.array(decisions)
    if set(y) != set(range(n_classes)):
        raise ValueError('Calibration diagnostics need true examples of all declared classes.')
    recall = {name:float(np.mean(prediction[y == i] == i)) for i, name in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes+1), dtype=int)
    for label, decision in zip(y, prediction):
        cm[label, decision if decision >= 0 else n_classes] += 1
    scored = prediction >= 0
    return dict(balanced_accuracy=float(np.mean(list(recall.values()))),
        accuracy=float(np.mean(y == prediction)), recall=recall,
        n_cues=len(y), n_scored=int(scored.sum()), n_rejected=int((~scored).sum()),
        coverage=float(scored.mean()), confusion_matrix=cm.tolist(),
        confusion_columns=classes+['rejected'], policy=resolved_policy, cues=details,
        scope='Clean extracted calibration windows, not full continuous validation.',
        unit_of_analysis='One majority deployed decision per cue; rejected cues count as errors.')
