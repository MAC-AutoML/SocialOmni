"""Validate the released annotations and recompute recorded primary metrics."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import sys

from metrics_core import binary_metrics, multiclass_metrics

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    return list(csv.DictReader((ROOT/name).open()))

def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-7), (a, b)

def main():
    level1 = json.loads((ROOT/'final_source/data/level_1/dataset.json').read_text())
    level2 = json.loads((ROOT/'final_source/data/level_2/annotations_200.json').read_text())['data']
    assert len(level1) == len({r['id'] for r in level1}) == 2000
    assert len(level2) == len({r['video_id'] for r in level2}) == 200
    assert {r['video_id'] for r in level2} == {f'video_{i:04d}' for i in range(1,201)}
    positive = {r['video_id'] for r in level2 if r['question_1']['correct_answer'] == 'A'}
    assert len(positive) == 128
    summary = rows('results/primary/model_summary.csv')
    assert len(summary) == 11
    scores = rows('results/primary/ensemble_response_scores.csv')
    assert len(scores) == len({(r['model'], r['item_id']) for r in scores}) == 688
    score_groups, when_groups = defaultdict(list), defaultdict(list)
    for r in scores:
        assert r['item_id'] in positive and r['candidate'].strip()
        values = [float(r['score_'+j]) for j in ('gpt4o','gemini_2_5_pro','qwen3_omni')]
        assert set(values) <= {0,25,50,75,100}
        close(sum(values)/3, r['q_ensemble'])
        score_groups[r['model']].append(r)
    for r in rows('results/when_predictions.csv'):
        r['error'] = 'request failure' if r['request_failed'] == 'True' else ''
        when_groups[r['model']].append(r)
    for s in summary:
        m = s['model']
        who = json.loads((ROOT/f'final_source/results/results_{m}_level1_audio-video_no-asr.json').read_text())['results']
        assert len(who) == 2000 and len(when_groups[m]) == 200
        for prefix, values in [('who',multiclass_metrics(who)),('when',binary_metrics(when_groups[m]))]:
            for metric in ('accuracy','macro_f1','failures','parse_failures'):
                close(values[metric], s[prefix+'_'+metric])
        generated = score_groups[m]
        close(len(generated), s['responses'])
        close(100*len(generated)/128, s['coverage_plus'])
        close(sum(float(r['q_ensemble']) for r in generated)/len(generated), s['q_ensemble'])
        close(sum(float(r['q_ensemble']) for r in generated)/128, s['q_joint_ensemble'])
    ext = {r['model']:r for r in rows('results/extension/main_extension_summary.csv')}
    for s in summary:
        for metric in ('who_accuracy','when_accuracy','responses','coverage_plus','q_ensemble','q_joint_ensemble'):
            close(s[metric],ext[s['model']][metric])
    ratings = rows('results/human/ratings.csv')
    keys = rows('results/human/calibration_key.csv')
    assert len(ratings) == len(keys) == 200
    assert {r['calibration_id'] for r in ratings} == {r['calibration_id'] for r in keys}
    assert len({r['model'] for r in keys}) == 10 and 'gpt4o' not in {r['model'] for r in keys}
    score_index = {(r['model'],r['item_id']) for r in scores}
    assert all((r['model'],r['item_id']) in score_index for r in keys)
    for r in ratings:
        assert {float(r['human'+str(i)+'_score']) for i in (1,2,3)} <= {0,25,50,75,100}
    manifest = ROOT/'SHA256SUMS.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print(json.dumps({'passed':True,'perception_items':2000,'interaction_items':200,
        'positive_states':128,'models':11,'responses':688,'primary_judge_scores':2064,
        'human_calibration_responses':200,'primary_metrics_recomputed':True},indent=2))

if __name__ == '__main__':
    main()
