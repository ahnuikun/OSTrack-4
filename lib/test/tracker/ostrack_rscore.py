"""Inference-only diagnostic: Rdisc scores, Tclean size/offset, native tracking.

Two complete forwards on the SAME tensors, not a production single-stream
architecture. No new training configuration or reliability/M2 policy.
"""

from pathlib import Path

import torch
from torch import nn

from lib.test.analysis.rscore_probe import PROBE_CONFIG
from tracking.diagnose_vdrm_frozen_failures import CONFIGS, config_for, load_models
from tracking.diagnose_vdrm_native_float import NativeTracker

class ScoreOnlyNetwork(nn.Module):
    def __init__(self, baseline, residual):
        super().__init__()
        self.baseline, self.residual = baseline, residual
        self.last_components = None
        self.requires_grad_(False).eval()

    @property
    def backbone(self):
        return self.residual.backbone

    @property
    def box_head(self):
        return self.baseline.box_head

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Rscore probe is inference-only; no training is authorized')
        return super().train(False)

    @torch.no_grad()
    def forward(self, *, template, search, ce_template_mask, template_bbox):
        if any(m.training for m in self.modules()) or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError('Rscore requires frozen parameters and eval mode')
        inputs = dict(template=template, search=search, ce_template_mask=ce_template_mask,
                      template_bbox=template_bbox)
        base = self.baseline.forward(**inputs)
        residual = self.residual.forward(**inputs)
        for key in ('score_map', 'size_map', 'offset_map'):
            if (base[key].shape != residual[key].shape or base[key].device != residual[key].device
                    or not torch.isfinite(base[key]).all() or not torch.isfinite(residual[key]).all()):
                raise RuntimeError('incompatible/non-finite Rscore maps: ' + key)
        if not torch.equal(base['search_global_index'], residual['search_global_index']):
            raise RuntimeError('pre-residual CE indices differ on identical probe inputs')
        output = dict(residual)
        output['size_map'], output['offset_map'] = base['size_map'], base['offset_map']
        # Native tracking decodes these maps again AFTER Hann. Keep the raw
        # pred_boxes field consistent too (no stale Rdisc regression boxes).
        output['pred_boxes'] = self.box_head.cal_bbox(
            output['score_map'], output['size_map'], output['offset_map']).view(-1, 1, 4)
        self.last_components = (base, residual)
        return output


class OSTrackRscore(NativeTracker):
    def __init__(self, params, dataset_name):
        if dataset_name != 'got10k_vdrm_dev':
            raise ValueError('Rscore probe is restricted to the fixed training development split')
        if params.debug or params.save_all_boxes:
            raise ValueError('Rscore diagnostic requires debug=0 and save_all_boxes=False')
        if params.cfg != config_for(CONFIGS['Rdisc']):
            raise ValueError('Rscore requires the unmodified original Rdisc configuration')
        if Path(params.checkpoint).resolve() != Path(params.rscore_paths['Rdisc']).resolve():
            raise ValueError('Rscore forbids an unregistered checkpoint override')
        models, _, self.provenance = load_models(params.rscore_paths, torch.device('cuda'),
                                                arms=('Tclean', 'Rdisc'))
        super().__init__(ScoreOnlyNetwork(models['Tclean'], models['Rdisc']), params.cfg)
        self.params = params
        print('Rscore probe: Rdisc score + Tclean size/offset; native Hann; two frozen forwards', flush=True)


def get_tracker_class():
    return OSTrackRscore
