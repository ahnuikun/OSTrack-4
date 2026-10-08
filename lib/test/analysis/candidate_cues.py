"""Rqual head-input interventions. All controls are offline, never a tracker."""
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np
import torch
from torch import nn

from lib.models.layers.candidate_quality import quality_delta, apply_energy_correction
from .response_probe import EARLY_PLAN, QUALITY_CONFIG
from .response_failure_audit import DECODE_TOLERANCE_PIXELS

ZERO_GROUPS = ('identity_zero', 'response_zero', 'geometry_zero', 'support_zero')
LOGIT_ATOL = 2e-5


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def read_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def build_corrector(state):
    required = {f'{layer}.{suffix}' for layer in (0, 2, 4) for suffix in ('weight', 'bias')}
    if set(state) != required:
        raise ValueError('unexpected candidate corrector parameters')
    model = nn.Sequential(nn.Linear(71,64), nn.GELU(), nn.Linear(64,64), nn.GELU(), nn.Linear(64,1))
    for name, value in model.state_dict().items():
        saved = state[name]
        if saved.shape != tuple(value.shape) or saved.dtype != np.float32 or not np.isfinite(saved).all():
            raise ValueError('invalid corrector tensor: '+name)
    model.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()}, strict=True)
    return model.requires_grad_(False).eval()


def intervention_cues(cues, candidates, donor):
    """Template-conditioned product/difference/cosine only; other cues fixed."""
    value = cues.clone()
    projected = candidates.shape[-1]
    value[..., :projected] = candidates * donor
    value[..., projected:2*projected] = (candidates - donor).abs()
    value[..., 2*projected:2*projected+1] = (candidates * donor).sum(-1, keepdim=True)
    return value


def variants(cues, candidates, donors):
    if cues.shape != (1,256,71) or candidates.shape != (1,256,32):
        raise ValueError('registered candidate cue dimensions required')
    output = {'native': cues}
    slices = {'identity_zero':slice(0,65), 'response_zero':slice(65,66),
              'geometry_zero':slice(66,70), 'support_zero':slice(70,71)}
    for name, group in slices.items():
        value = cues.clone()
        value[..., group] = 0
        output[name] = value
    for name, prototype in sorted(donors.items()):
        output['swap:'+name] = intervention_cues(cues, candidates, prototype)
    return output


@torch.no_grad()
def evaluate_controls(head, cues, candidates, donors, valid, base):
    outputs = {}
    for name, value in variants(cues, candidates, donors).items():
        logits = head(value).squeeze(-1)
        delta = quality_delta(logits, valid, 2.)
        score = apply_energy_correction(base, delta)
        outputs[name] = dict(logits=logits, score=score)
    return outputs


def frame_metrics(record, arrays):
    boxes = arrays['boxes_xywh'].astype(np.float64)
    gt = np.asarray(record['gt_xywh'], dtype=np.float64)
    valid_gt = bool((gt[2:] > 0).all())
    if valid_gt:
        intersection = np.maximum(np.minimum(boxes[:,:2]+boxes[:,2:],gt[:2]+gt[2:]) -
                                  np.maximum(boxes[:,:2],gt[:2]),0).prod(1)
        ious = intersection / np.maximum(boxes[:,2:].prod(1)+gt[2:].prod()-intersection,1e-12)
    else:
        ious = np.full(256,np.nan)
    hann = arrays['hann'].reshape(256)
    scores, logits = arrays['scores'].reshape(-1,256), arrays['logits'].reshape(-1,256)
    peaks = (scores*hann).argmax(1)
    baseline_peak = int((arrays['base_score'].reshape(256)*hann).argmax())
    selected = arrays['valid'].reshape(256)
    result = dict(sequence=record['sequence'], frame_1based=record['frame_1based'],
        visible=bool(valid_gt and record['cover'] > 0 and record['absence']==0),
        base_peak=baseline_peak, base_iou=float(ious[baseline_peak]) if valid_gt else None,
        native_peak=int(peaks[0]), native_iou=float(ious[peaks[0]]) if valid_gt else None, controls={})
    for index,name in enumerate(record['variants']):
        result['controls'][name]=dict(peak=int(peaks[index]), changed_peak=bool(peaks[index]!=peaks[0]),
            iou=float(ious[peaks[index]]) if valid_gt else None,
            mean_abs_logit_change=float(np.abs(logits[index,selected]-logits[0,selected]).mean()) if selected.any() else 0.,
            quality_at_native=float(torch.tensor(logits[index,peaks[0]]).sigmoid()),
            quality_at_base=float(torch.tensor(logits[index,baseline_peak]).sigmoid()))
    return result


def check_native(record, arrays, metric):
    """Ensure the unmodified exported head still represents the source tracker."""
    actual, expected = metric['native_iou'], record['source_native_iou']
    iou_changed = (actual is None) != (expected is None)
    if actual is not None and expected is not None:
        iou_changed |= abs(actual - expected) > 1e-4
    box = arrays['boxes_xywh'][metric['native_peak']]
    if (iou_changed or metric['native_peak'] != record['source_native_peak'] or
            not np.allclose(box, record['source_native_bbox'], rtol=0, atol=DECODE_TOLERANCE_PIXELS)):
        raise ValueError('native candidate differs from source trajectory')


def summarize(frames):
    groups = {}
    for row in frames:
        for name,value in row['controls'].items():
            key = 'template_swap' if name.startswith('swap:') else name
            groups.setdefault(key,[]).append((row,value))
    report = {}
    for name,rows in groups.items():
        visible=[(r,v) for r,v in rows if r['visible']]
        report[name]=dict(interventions=len(rows), changed_peaks=sum(v['changed_peak'] for _,v in rows),
            mean_abs_logit_change=float(np.mean([v['mean_abs_logit_change'] for _,v in rows])),
            visible_comparisons=len(visible),
            improved_gt5pp=sum(v['iou']>r['native_iou']+.05 for r,v in visible),
            worsened_gt5pp=sum(v['iou']<r['native_iou']-.05 for r,v in visible))
    return dict(records=len(frames), groups=report, optimizer_steps=0,
        closed_loop=False, automatic_architecture_verdict=False,
        note='Four donors per frame are correlated interventions, not independent samples. '
             'Zeroing cues is out of distribution; this measures head dependence, not deployment performance.')


def render_report(summary):
    lines=['# Rqual candidate-cue audit','',
        'Fixed Rqual crops/boxes; current GT is evaluation-only. No training or state updates.',
        'Input ablations are out-of-distribution diagnostics, NOT proposed deployment switches.', '',
        '| Control | Interventions | Peak changes | Mean absolute logit change | Visible +5pp / -5pp |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,row in summary['groups'].items():
        lines.append(f"| {name} | {row['interventions']} | {row['changed_peaks']} | "
            f"{row['mean_abs_logit_change']:.6f} | {row['improved_gt5pp']} / {row['worsened_gt5pp']} |")
    lines += ['',summary['note'],'',
        'Use per_frame.json to inspect 003240:93, 001664:74/75, 007878:22/70/73 and '
        '008311:39/40/43. Do not infer a closed-loop gain from a single-step control.','']
    return '\n'.join(lines)


def check_bundle(path):
    """CPU recomputation of exported MLP, every intervention and report."""
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or
            '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members):
            raise ValueError('unsafe/duplicate candidate-cue members')
        payloads={m.name:archive.extractfile(m).read() for m in members}
    manifest=json.loads(payloads.pop('manifest.json'))
    if (manifest.get('kind')!='vdrm_candidate_cues' or manifest.get('version')!=1 or
            manifest.get('dataset')!='got10k_vdrm_dev' or manifest.get('config')!=QUALITY_CONFIG or manifest.get('optimizer_steps')!=0 or
            manifest.get('records')!=298 or manifest.get('state_sha256_before')!=manifest.get('state_sha256_after')):
        raise ValueError('wrong candidate-cue contract')
    expected={(s,f) for s,(a,b) in EARLY_PLAN.items() for f in range(a,b+1)}
    if manifest['members_sha256']!={k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()}:
        raise ValueError('candidate-cue checksum mismatch')
    records=json.loads(payloads['records.json'])
    keys=[(r['sequence'],r['frame_1based']) for r in records]
    if len(keys)!=298 or len(set(keys))!=298 or set(keys)!=expected:
        raise ValueError('incomplete candidate-cue frames')
    frame_members={r['member'] for r in records}
    if len(frame_members)!=298 or set(payloads)!=frame_members | {
            'head.npz','donors.npz','records.json','per_frame.json','summary.json','report.md'}:
        raise ValueError('unexpected/missing candidate-cue members')
    head=build_corrector(read_npz(payloads['head.npz']))
    donors=read_npz(payloads['donors.npz'])
    if set(donors)!=set(EARLY_PLAN) or any(v.shape!=(1,1,32) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in donors.values()):
        raise ValueError('invalid donor set')
    frames=[]
    with torch.no_grad():
        for row in records:
            arrays=read_npz(payloads[row['member']])
            shapes=dict(cues=(1,256,71),candidate_embedding=(1,256,32),template_embedding=(1,1,32),
                valid=(1,256),base_score=(1,1,16,16),hann=(1,1,16,16),boxes_xywh=(256,4),
                logits=(9,1,256),scores=(9,1,1,16,16))
            if set(arrays)!=set(shapes) or any(arrays[k].shape!=s or not np.isfinite(arrays[k]).all() for k,s in shapes.items()):
                raise ValueError('invalid exported candidate features')
            if arrays['valid'].dtype!=np.bool_:
                raise ValueError('candidate validity must be boolean')
            if any(arrays[k].dtype!=np.float32 for k in shapes if k not in ('valid','boxes_xywh')):
                raise ValueError('candidate cues and scores must be float32')
            if (arrays['boxes_xywh'][:,2:]<=0).any():
                raise ValueError('decoded boxes must have positive sizes')
            if row['frame_1based']==EARLY_PLAN[row['sequence']][0] and not np.array_equal(
                    arrays['template_embedding'],donors[row['sequence']]):
                raise ValueError('donor differs from the first saved frame')
            tensors={k:torch.from_numpy(v) for k,v in arrays.items()}
            reconstructed=intervention_cues(tensors['cues'],tensors['candidate_embedding'],tensors['template_embedding'])
            if not torch.allclose(reconstructed,tensors['cues'],rtol=0,atol=2e-6):
                raise ValueError('identity cues differ from exported embeddings')
            controls=evaluate_controls(head,tensors['cues'],tensors['candidate_embedding'],
                {s:torch.from_numpy(v) for s,v in donors.items() if s!=row['sequence']},tensors['valid'],tensors['base_score'])
            if list(controls)!=row['variants']:
                raise ValueError('wrong intervention order/donor identity')
            for index,(name,value) in enumerate(controls.items()):
                if not torch.allclose(value['logits'],tensors['logits'][index],rtol=1e-5,atol=LOGIT_ATOL):
                    raise ValueError('exported MLP does not reproduce intervention: '+name)
                # Recompute energy from STORED logits, separating CPU/GPU GEMM
                # roundoff from score-formula validation. Peaks use saved scores.
                delta=quality_delta(tensors['logits'][index],tensors['valid'],2.)
                score=apply_energy_correction(tensors['base_score'],delta)
                if not torch.allclose(score,tensors['scores'][index],rtol=2e-6,atol=2e-7):
                    raise ValueError('intervention energy score formula differs')
            metric=frame_metrics(row,arrays)
            check_native(row, arrays, metric)
            frames.append(metric)
    summary=summarize(frames)
    if (frames!=json.loads(payloads['per_frame.json']) or summary!=json.loads(payloads['summary.json'])
            or render_report(summary)!=payloads['report.md'].decode('utf-8')):
        raise ValueError('candidate-cue report differs from independent recomputation')
    return manifest,summary,frames
