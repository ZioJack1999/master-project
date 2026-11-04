"""Low-rank deployment must preserve inference and exclude target covariances."""
import copy
import numpy as np
import pytest
import torch
from multiclass_features import FeatureNetwork
from experiments.multiclass_v4.lowrank import fit_projection, project, makehead, fold_state_dict
from experiments.multiclass_v4.study import (SourceCovarianceGuard, development_indices,
                                             fit_case)

torch.set_num_threads(1)


def records_features(n_cues=21, dimensions=80):
    rng = np.random.default_rng(14)
    rows = [dict(trial=f'cue{i:02d}',label=i%3) for i in range(n_cues) for _ in range(2)]
    x = rng.normal(size=(len(rows),dimensions)).astype(np.float32)
    return x, rows


def test_projection_counts_whole_cues_and_is_invariant_to_overlapping_duplicates():
    x, rows = records_features()
    p = fit_projection(x,rows,8)
    repeat = np.array([i for i,r in enumerate(rows) if r['label'] == 0])
    q = fit_projection(np.concatenate([x,x[repeat]]),rows+[rows[i] for i in repeat],8)
    assert p['n_training_cues'] == 21 and not p['whitening']
    np.testing.assert_allclose(p['center'],q['center'],atol=1e-7)
    # SVD component signs may differ, so compare the retained subspace.
    np.testing.assert_allclose(p['components'].T@p['components'],q['components'].T@q['components'],atol=1e-6)
    np.testing.assert_allclose(p['components']@p['components'].T,np.eye(8),atol=1e-6)


def test_projection_caps_independent_rank_and_rejects_conflicting_cues():
    x, rows = records_features(n_cues=6)
    p = fit_projection(x,rows,16)
    assert p['effective_components'] <= 5
    bad = copy.deepcopy(rows);bad[0]['label'] = 2
    with pytest.raises(ValueError,match='conflicting'):
        fit_projection(x,bad,8)
    with pytest.raises(ValueError,match='No between-cue'):
        fit_projection(np.ones_like(x),rows,8)


@pytest.mark.parametrize('components',[8,16])
def test_folded_existing_network_matches_projected_head_without_mutation(components):
    x, rows = records_features(dimensions=840)
    projection = fit_projection(x,rows,components)
    torch.manual_seed(5)
    head = makehead(components,16,3,.5).eval()
    before = copy.deepcopy(head.state_dict())
    folded = fold_state_dict(head.state_dict(),projection)
    deployed = FeatureNetwork('log_covariance',20,3,{},hidden=16,dropout=.5).eval()
    deployed.load_state_dict(folded)
    with torch.inference_mode():
        a = torch.softmax(head(torch.from_numpy(project(x,projection))),1).numpy()
        b = torch.softmax(deployed(torch.from_numpy(x)),1).numpy()
    np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    for key,value in head.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_source_guard_rejects_test_rows_and_overlapping_sessions():
    rows = [dict(subject='08',session=session,label_name=name,trial=f'{session}_{name}')
            for session in [1,2,3] for name in ['rest','volar_grasp','finger_extension']]
    idx,y = development_indices(rows,'08',['rest','volar_grasp','finger_extension'],[1],[2])
    guard = SourceCovarianceGuard(np.arange(9),set(idx['train'])|set(idx['validation']))
    np.testing.assert_array_equal(guard[idx['train']],np.arange(3))
    with pytest.raises(ValueError,match='test/undeclared'):
        guard[np.array([6])]
    with pytest.raises(ValueError,match='disjoint'):
        development_indices(rows,'08',['rest','volar_grasp','finger_extension'],[1],[1])


def test_case_fits_and_exports_without_ever_reading_test_covariances(tmp_path):
    classes=['rest','volar_grasp','finger_extension']
    records = [dict(subject='08',session=session,label_name=name,
                    trial=f'{session}_{label}_{cue}',run=f'{session}_{cue}')
               for session in [1,2,3] for label,name in enumerate(classes) for cue in range(2)]
    idx,y = development_indices(records,'08',classes,[1],[2])
    rng=np.random.default_rng(21)
    a=rng.normal(size=(len(records),4,20,24))
    cov=(a@a.swapaxes(-1,-2)*1e-12).astype(np.float32)
    cov[12:]=np.nan
    allowed=set(idx['train'])|set(idx['validation'])
    guard=SourceCovarianceGuard(cov,allowed)
    case=dict(name='synthetic_source_only',subject='08')
    candidate=dict(name='pca8_hidden16',components=8,hidden=16,dropout=.5,weight_decay=.05)
    config=dict(covariance_shrinkage=.1,seeds=[42],epochs=2,patience=2,
                windows_per_cue=4,batch_size=16,learning_rate=.001,
                experiment_id='synthetic_source_guard')
    result=fit_case(case,candidate,guard,records,idx,y,classes,
                    [f'EEG{i}' for i in range(20)],config,tmp_path)
    assert guard.read_indices <= allowed
    assert result['no_test_covariance_access'] and not result['test_scored']
    assert result['projection']['effective_components'] <= 5
    assert max(result['folding_max_probability_errors']) < 1e-5
    assert (tmp_path/'model.pt.json').exists()
