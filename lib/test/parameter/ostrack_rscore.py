"""A TEST-ONLY alias; load original source configs without changing contracts."""

from pathlib import Path

from lib.test.evaluation.environment import env_settings
from lib.test.parameter.ostrack import parameters as native_parameters
from lib.test.tracker.ostrack_rscore import CONFIGS, PROBE_CONFIG


def parameters(name):
    if name != PROBE_CONFIG:
        raise ValueError('ostrack_rscore accepts only the registered Rscore probe name')
    params = native_parameters(CONFIGS['Rdisc'])
    root = Path(env_settings().save_dir) / 'checkpoints/train/ostrack'
    params.rscore_paths = {a: root / CONFIGS[a] / 'OSTrack_ep0300.pth.tar'
                          for a in ('Tclean', 'Rdisc')}
    for path in params.rscore_paths.values():
        if not path.is_file():
            raise FileNotFoundError('missing Rscore source checkpoint: ' + str(path))
    return params
