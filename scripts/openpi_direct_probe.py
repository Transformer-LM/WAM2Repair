import json, pathlib, traceback, time
from openpi.training import config as cfgmod
from openpi.policies import policy_config

out = pathlib.Path('__WAM2REPAIR_ROOT__/logs/wam2repair/openpi/direct_probe.json')
out.parent.mkdir(parents=True, exist_ok=True)
result = {'stage': 'start', 'time': time.time()}
try:
    cfg = cfgmod.get_config('pi05_libero')
    result['config_action_dim'] = cfg.model.action_dim
    result['config_action_horizon'] = cfg.model.action_horizon
    policy = policy_config.create_trained_policy(cfg, '__PI05_CHECKPOINT__')
    result['stage'] = 'policy_created'
    result['metadata'] = getattr(policy, 'metadata', {})
except Exception as e:
    result['stage'] = 'error'
    result['error'] = repr(e)
    result['traceback'] = traceback.format_exc()
out.write_text(json.dumps(result, indent=2, default=str))
print(json.dumps(result, indent=2, default=str))
