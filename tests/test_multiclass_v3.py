"""Leakage, numerical and binary-separation checks for experimental research."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.multiclass_v3.models import (Features, Runner, CueNetwork,
    trial_weights, alignment_reference, align, fit)
from experiments.multiclass_v3.study import (source_views, adaptation_partition,
    unlabeled_prefix, search_case, verify_frozen, personal_indices)
from create_personal_template import MONTAGES
from realtime_bci_decoder import load_contract
from verify_binary_release import verify

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['rest', 'volar_grasp', 'finger_extension']


def fixture():
    rows = []
    for subject in ['04', '08', '07']:
        for session in [1, 2, 3]:
            for label in range(3):
                for window in range(2):
                    rows.append(dict(subject=subject, session=session, label=label,
                        label_name=CLASSES[label], run=f'{subject}_{session}',
                        trial=f'{subject}_{session}_{label}'))
    rng = np.random.default_rng(5)
    a = rng.normal(size=(len(rows), 4, 20, 22))
    cov = (a@a.swapaxes(-1, -2)*1e-12).astype(np.float32)
    idx = {p:np.array([i for i, r in enumerate(rows) if
           (r['subject'] == '07' if p == 'test' else
            r['subject'] != '07' and (r['session'] == 3 if p == 'validation' else r['session'] != 3))])
           for p in ['train', 'validation', 'test']}
    return cov, rows, idx, np.array([r['label'] for r in rows])


def test_cue_weighting_does_not_count_overlapping_copies_as_new_trials():
    _, rows, _, _ = fixture()
    groups, labels, weights = trial_weights(rows)
    duplicate = rows+[dict(r) for r in rows if r['subject'] == '04']
    dg, dy, dw = trial_weights(duplicate)
    assert len(groups) == len(dg)
    np.testing.assert_array_equal(labels, dy)
    np.testing.assert_allclose(weights, dw)
    corrupt = copy.deepcopy(rows)
    corrupt[0]['label'] = 2
    with pytest.raises(ValueError, match='one participant and one label'):
        trial_weights(corrupt)


def test_alignment_handles_physical_volts_and_corrects_known_spatial_scale():
    ref = np.diag(np.linspace(1, 5, 20))*1e-11
    cov = np.tile(ref[None, None], (6, 4, 1, 1))
    rows = [dict(trial=str(i), subject='07', label=0) for i in range(6)]
    transform = alignment_reference(cov, rows)
    actual = align(cov, transform).sum(axis=1)
    np.testing.assert_allclose(actual, np.tile(np.eye(20)[None], (6, 1, 1)), atol=1e-6)
    changed_labels = [dict(r, label=i%3) for i, r in enumerate(rows)]
    np.testing.assert_array_equal(transform, alignment_reference(cov, changed_labels))


def test_alignment_prefix_is_earlier_disjoint_and_not_class_filtered():
    _, rows, idx, _ = fixture()
    prefix, future = adaptation_partition(idx['test'], rows, [1])
    assert not set(prefix)&set(future)
    # A gesture outside a three-class task must still be included unlabeled.
    rows.append(dict(subject='07', session=1, trial='other', label=8))
    assert len(unlabeled_prefix(rows, '07', [1])) == len(prefix)+1
    with pytest.raises(ValueError, match='not earlier'):
        adaptation_partition(idx['test'], rows, [2])


def test_source_views_do_not_read_target_and_alignment_does_not_fit_validation():
    cov, rows, idx, y = fixture()
    class Guard:
        def __getitem__(self, indices):
            assert not set(indices)&set(idx['test'])
            return cov[indices]
    _, _, _, _, refs = source_views(Guard(), rows, idx, y, True)
    cov[idx['validation']] *= 1e10
    _, _, _, _, after = source_views(Guard(), rows, idx, y, True)
    for subject in refs:
        np.testing.assert_array_equal(refs[subject], after[subject])


def test_personal_sessions_and_people_remain_separate():
    _, rows, _, _ = fixture()
    split = dict(train=[1], validation=[2], test=[3])
    idx, y = personal_indices(rows, CLASSES, '07', split)
    for part, indices in idx.items():
        assert {rows[i]['subject'] for i in indices} == {'07'}
        assert {rows[i]['session'] for i in indices} == set(split[part])
        assert set(y[indices]) == {0, 1, 2}
    with pytest.raises(ValueError, match='overlap'):
        personal_indices(rows, CLASSES, '07', dict(split, test=[1]))


@pytest.mark.parametrize('name', ['power_motor', 'relative_motor', 'tangent_motor', 'tangent_all'])
def test_feature_and_model_round_trip_without_refitting(name, tmp_path):
    cov, rows, idx, _ = fixture()
    tr = idx['train']
    feature = Features.fit(cov[tr], [rows[i] for i in tr], MONTAGES['x24_stock20'], name)
    x = feature.transform(cov)
    torch.manual_seed(3)
    model = CueNetwork(x.shape[1], 3, 'small_nn')
    payload = dict(features=feature.parameters, classes=CLASSES, inputs=x.shape[1],
                   head='small_nn', hidden_units=16, dropout=.5, states=[model.state_dict()])
    path = tmp_path/'model.pt'
    torch.save(payload, path)
    a = Runner(payload).predict(cov)
    b = Runner(torch.load(path, weights_only=False)).predict(cov)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-6)


def test_search_is_frozen_and_binary_loader_rejects_research(tmp_path):
    cov, rows, idx, y = fixture()
    c = dict(feature_sets=['power_motor'], heads=['linear'], covariance_shrinkage=.3,
             selection_seed=42, ensemble_seeds=[42], hidden_units=16, dropout=.5,
             learning_rate=.001, weight_decay=.05, label_smoothing=.05,
             epochs=2, patience=2, batch_cues=8, windows_per_cue=4)
    hashes = dict(synthetic=True)
    search_case(tmp_path, cov, rows, CLASSES, idx, y, False,
                dict(channels=MONTAGES['x24_stock20']), c, hashes)
    before = (tmp_path/'model.pt').read_bytes()
    verify_frozen(tmp_path, hashes)
    search_case(tmp_path, cov, rows, CLASSES, idx, y, False,
                dict(channels=MONTAGES['x24_stock20']), c, hashes)
    assert before == (tmp_path/'model.pt').read_bytes()
    contract = json.loads((tmp_path/'model.pt.json').read_text())
    # Supply binary loader's required fields so class rejection is exercised.
    contract.update(normalization='fixed')
    (tmp_path/'model.pt.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='Expected output classes'):
        load_contract(tmp_path/'model.pt')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_frozen(tmp_path, hashes)


def test_binary_release_matches_frozen_files_and_detects_changes(tmp_path):
    result = verify(ROOT)
    assert result['binary_release_unchanged'] and result['files_checked'] == 36
    manifest = json.loads((ROOT/'binary_release_manifest.json').read_text())
    assert 'studio.html' in manifest['sha256']
    for relative in manifest['sha256']:
        dest = tmp_path/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT/relative).read_bytes()
        if relative in manifest.get('text_lf_sha256', {}):
            # A fresh checkout can use either line ending across platforms.
            data = data.replace(b'\r\n', b'\n')
        dest.write_bytes(data)
    assert verify(tmp_path, manifest)['binary_release_unchanged']
    (tmp_path/'networks.py').write_text('changed')
    result = verify(tmp_path, manifest)
    assert not result['binary_release_unchanged']
    assert result['problems'] == [dict(file='networks.py', issue='changed')]
