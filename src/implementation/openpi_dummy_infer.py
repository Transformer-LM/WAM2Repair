"""Offline one-request OpenPI inference probe; never connects to a robot."""
import json
import pathlib
import time
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

def main():
    out = pathlib.Path('__WAM2REPAIR_ROOT__/results/wam2repair/e1/openpi_dummy_infer.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    obs = {
        'observation/state': np.zeros((8,), dtype=np.float32),
        'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
        'prompt': 'move the object to the target',
    }
    with connect('ws://127.0.0.1:10098', compression=None, max_size=None,
                 ping_interval=None, open_timeout=60, close_timeout=5) as conn:
        metadata = msgpack_numpy.unpackb(conn.recv(timeout=60))
        rows = []
        for repeat in range(2):
            t0 = time.monotonic()
            conn.send(msgpack_numpy.Packer().pack(obs))
            response = conn.recv(timeout=1200)
            if isinstance(response, str):
                raise RuntimeError(response)
            result = msgpack_numpy.unpackb(response)
            elapsed = time.monotonic() - t0
            actions = np.asarray(result['actions'])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError(f'Invalid actions: shape={actions.shape}')
            rows.append({'repeat': repeat, 'infer_ms': elapsed * 1000,
                         'server_timing': result.get('server_timing')})
    payload = {
        'schema': 'wam2repair-openpi-dummy-infer-v1',
        'status': 'PASS',
        'server_metadata': metadata,
        'claim_scope': 'dummy input transport and numeric sanity only; no task performance',
        'repeats': rows,
        'result_keys': sorted(result.keys()),
        'actions_shape': list(actions.shape),
        'actions_finite': bool(np.isfinite(actions).all()),
        'actions_first': actions[0].tolist() if actions.ndim > 1 else actions.tolist(),
        'infer_ms': elapsed * 1000.0,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(json.dumps(payload, indent=2, default=str))

if __name__ == '__main__':
    main()
