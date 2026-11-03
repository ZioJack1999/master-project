"""Offline multiclass search, freeze, then one retrospective score per case.

Never imports the GUI, acquisition or transport. Nothing writes binary models.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from multiclass_data import digest
from multiclass_cohort_data import load_cohort
from multiclass_metrics import evaluate
from study_multiclass_transfer import case_indices, rows_for
from train_multiclass import dump
from .models import Features, Runner, fit, diagnostics, selection_score, alignment_reference, align

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def declared_cases(records, base, c):
    cases = []
    for personal in c.get('personal_cases', []):
        for names in personal['class_sets']:
            classes = base['class_sets'][names]
            idx, y = personal_indices(records, classes, personal['subject'], personal['split'])
            cases.append((f"personal_{personal['subject']}_{names}", classes, idx, y, False))
    for spec in c['within_cases']:
        cohort, names = spec.split('_')
        classes = base['class_sets'][names]
        idx, y = case_indices(records, classes, base, within=base['within_person_splits'][cohort])
        cases.append((f'within_08_{spec}', classes, idx, y, False))
    for names in c['classes']:
        classes = base['class_sets'][names]
        idx, y = case_indices(records, classes, base, held_subject=c['transfer_held_id'])
        cases.append((f'transfer_hold_07_{names}', classes, idx, y, False))
        cases.append((f'prefix_ea_hold_07_{names}', classes, idx, y, True))
    return cases


def personal_indices(records, classes, subject, split):
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('Personal experiment requires three explicit partitions.')
    sessions = [set(split[p]) for p in ['train', 'validation', 'test']]
    if any(a & b for a, b in [(sessions[0], sessions[1]), (sessions[0], sessions[2]), (sessions[1], sessions[2])]):
        raise ValueError('Personal sessions overlap.')
    mapped = np.array([classes.index(r['label_name']) if r['label_name'] in classes else -1 for r in records])
    idx = {p: np.array([i for i, r in enumerate(records) if
                       r['subject'] == subject and r['session'] in split[p] and mapped[i] >= 0], dtype=int)
           for p in split}
    for p, indices in idx.items():
        if set(mapped[indices]) != set(range(len(classes))):
            raise ValueError(f'Personal {p} lacks classes.')
    return idx, mapped


def source_views(cov, records, idx, y, ea):
    # This function is deliberately unable to read any test index.
    train, val = idx['train'], idx['validation']
    a, b = np.asarray(cov[train]).copy(), np.asarray(cov[val]).copy()
    tr, vr = rows_for(records, train, y), rows_for(records, val, y)
    refs = {}
    if ea:
        for subject in sorted({r['subject'] for r in tr}):
            ti = [i for i, r in enumerate(tr) if r['subject'] == subject]
            vi = [i for i, r in enumerate(vr) if r['subject'] == subject]
            ref = alignment_reference(a[ti], [tr[i] for i in ti])
            refs[subject] = ref
            a[ti] = align(a[ti], ref)
            b[vi] = align(b[vi], ref)
    return a, b, tr, vr, refs


def search_case(folder, cov, records, classes, idx, y, ea, manifest, config, hashes):
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder/'frozen.json'
    if frozen.exists():
        verify_frozen(folder, hashes)
        return
    if (folder/'results.json').exists():
        raise ValueError('Scored experiment cannot be searched again.')
    a, b, tr, vr, refs = source_views(cov, records, idx, y, ea)
    results = []
    best = None
    for name in config['feature_sets']:
        features = Features.fit(a, tr, manifest['channels'], name, config['covariance_shrinkage'])
        x, vx = features.transform(a), features.transform(b)
        for head in config['heads']:
            model, history = fit(x, tr, vx, vr, classes, head, config, config['selection_seed'])
            candidate = dict(feature_set=name, head=head, inputs=x.shape[1], diagnostics=history)
            results.append(candidate)
            score = selection_score(history['validation'])
            print(folder.name, name, head, 'train %.3f val %.3f' %
                  (history['train']['mean_person_cue_ba'], score[0]), flush=True)
            if best is None or score > best['score']:
                best = dict(score=score, features=copy.deepcopy(features.parameters), candidate=candidate,
                            x=x, vx=vx, selected_state=copy.deepcopy(model.state_dict()))
    # Fixed three-seed ensemble of the validation-selected candidate.
    states, histories = [], []
    head = best['candidate']['head']
    for seed in config['ensemble_seeds']:
        if seed == config['selection_seed']:
            states.append(best['selected_state'])
            histories.append(best['candidate']['diagnostics'])
        else:
            model, history = fit(best['x'], tr, best['vx'], vr, classes, head, config, seed)
            states.append(copy.deepcopy(model.state_dict()))
            histories.append(history)
    payload = dict(arch='offline_cue_bag_ensemble', task_mode='multiclass', classes=classes,
                   features=best['features'], states=states, inputs=best['x'].shape[1], head=head,
                   hidden_units=config['hidden_units'], dropout=config['dropout'],
                   source_alignment_references=refs, prefix_ea=ea)
    model_path = folder/'model.pt'
    torch.save(payload, model_path)
    runner = Runner(payload)
    development = dict(candidates=results, selected_feature=best['candidate']['feature_set'], head=head,
                       ensemble_histories=histories, train=diagnostics(runner.predict(a), tr, classes),
                       validation=diagnostics(runner.predict(b), vr, classes),
                       classes=classes, alignment=ea, test_scored=False)
    dump(folder/'development.json', development)
    contract = dict(task_mode='multiclass', role='offline_exploratory_only', classes=classes,
                    sha256=digest(model_path), arch=payload['arch'], channels=manifest['channels'],
                    feature_channels=best['features']['channels'], sfreq=256, n_times=1024,
                    filter=dict(mode='causal', fmin=8., fmax=30., order=4),
                    policy_status='No live GUI or actuator integration; never select in binary Studio.',
                    prefix_ea=ea, verified=False,
                    exposure='Previously exposed historical sessions; not confirmatory validation.')
    dump(str(model_path)+'.json', contract)
    dump(frozen, dict(model_sha256=digest(model_path), contract_sha256=digest(str(model_path)+'.json'),
                      development_sha256=digest(folder/'development.json'), data_hashes=hashes,
                      selected_feature=development['selected_feature'], head=head,
                      target_used_for_selection=False, timestamp=time.time()))


def verify_frozen(folder, hashes):
    f = load_config(folder/'frozen.json')
    for field, path in [('model_sha256', folder/'model.pt'),
                        ('contract_sha256', folder/'model.pt.json'),
                        ('development_sha256', folder/'development.json')]:
        if f[field] != digest(path):
            raise ValueError('Frozen research artifact changed.')
    if f['data_hashes'] != hashes:
        raise ValueError('Frozen experiment input changed.')


def adaptation_partition(indices, records, prefix_sessions):
    prefix = np.array([i for i in indices if records[i]['session'] in prefix_sessions], dtype=int)
    future = np.array([i for i in indices if records[i]['session'] not in prefix_sessions], dtype=int)
    if not len(prefix) or not len(future):
        raise ValueError('Adaptation requires a nonempty earlier prefix and later evaluation.')
    if max(records[i]['session'] for i in prefix) >= min(records[i]['session'] for i in future):
        raise ValueError('Adaptation prefix is not earlier than every evaluated session.')
    if {records[i]['trial'] for i in prefix} & {records[i]['trial'] for i in future}:
        raise ValueError('Adaptation cue leaked into scoring.')
    return prefix, future


def unlabeled_prefix(records, subject, sessions):
    # Include ALL accepted gesture windows, even gestures outside the model's
    # class set. Selecting only the scored classes would require target labels.
    return np.array([i for i, r in enumerate(records)
                     if r['subject'] == subject and r['session'] in sessions], dtype=int)


def score_case(folder, cov, records, classes, idx, y, ea, config, hashes):
    verify_frozen(folder, hashes)
    result_path = folder/'results.json'
    if result_path.exists():
        return load_config(result_path)
    payload = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    runner = Runner(payload)
    test = idx['test']
    adaptation = None
    if ea:
        prefix, test = adaptation_partition(test, records, config['target_prefix_sessions'])
        prefix = unlabeled_prefix(records, config['transfer_held_id'], config['target_prefix_sessions'])
        # Remove labels before computing the unlabeled prefix reference.
        prefix_rows = [dict(records[i], label=0) for i in prefix]
        ref = alignment_reference(np.asarray(cov[prefix]), prefix_rows)
        target_cov = align(np.asarray(cov[test]), ref)
        torch.save(dict(reference=ref, prefix_sessions=config['target_prefix_sessions']), folder/'target_prefix.pt')
        adaptation = dict(prefix_sessions=config['target_prefix_sessions'],
                          evaluated_sessions=sorted({records[i]['session'] for i in test}),
                          prefix_cues=len({records[i]['trial'] for i in prefix}),
                          reference_sha256=digest(folder/'target_prefix.pt'), labels_used=False)
        # Save the adapted reference BEFORE any later-session prediction.
        dump(folder/'adaptation_frozen.json', adaptation)
    else:
        target_cov = np.asarray(cov[test])
    rows = rows_for(records, test, y)
    p = runner.predict(target_cov)
    result = dict(case=folder.name, classes=classes, development=load_config(folder/'development.json'),
                  historical_test=evaluate(p, y[test], rows, classes, uncertainty=True),
                  adaptation=adaptation, model_sha256=digest(folder/'model.pt'),
                  interpretation='Exploratory historical evaluation, not a fresh or prospective test.')
    # Fair comparison: nonadapted model evaluated on the SAME later-session subset.
    if not ea and folder.name.startswith('transfer'):
        _, later = adaptation_partition(test, records, config['target_prefix_sessions'])
        positions = np.flatnonzero(np.isin(test, later))
        result['same_later_sessions_without_adaptation'] = evaluate(
            p[positions], y[later], rows_for(records, later, y), classes)
    dump(result_path, result)
    print('HISTORICAL SCORE', folder.name, result['historical_test']['cue_trials']['balanced_accuracy'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['search', 'score'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('experiment.json'))
    args = parser.parse_args()
    c = load_config(args.config)
    torch.set_num_threads(c['threads'])
    cohort, work = ROOT/c['cohort'], ROOT/c['output']
    cov, records, manifest = load_cohort(cohort)
    hashes = {name: digest(cohort/name) for name in ['cohort_manifest.json', 'records.json', 'covariances.npy']}
    hashes['experiment'] = digest(args.config)
    cases = declared_cases(records, manifest['config'], c)
    if args.stage == 'search':
        if (work/'all_selections_frozen.json').exists():
            raise ValueError('All selections are frozen; use score, or new independent development data.')
        for name, classes, idx, y, ea in cases:
            print('SEARCH', name, flush=True)
            search_case(work/name, cov, records, classes, idx, y, ea, manifest, c, hashes)
        dump(work/'all_selections_frozen.json', {name: digest(work/name/'frozen.json') for name, *_ in cases})
    else:
        sealed = load_config(work/'all_selections_frozen.json')
        for name, *_ in cases:
            if sealed[name] != digest(work/name/'frozen.json'):
                raise ValueError('Selection set changed after freezing.')
            verify_frozen(work/name, hashes)
        results = [score_case(work/name, cov, records, classes, idx, y, ea, c, hashes)
                   for name, classes, idx, y, ea in cases]
        dump(work/'study_results.json', dict(config=c, results=results,
             warning='Historical data are exposed. No new confirmatory accuracy claim. Binary release unchanged.'))


if __name__ == '__main__':
    main()
