"""Personal pi0.5 websocket server that requires a client-provided diffusion noise tensor."""
import argparse
import asyncio
import http
import logging
import socket
import traceback
import hashlib
import json
from pathlib import Path

import numpy as np
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames
from openpi_client import msgpack_numpy
from openpi.policies import policy_config
from openpi.training import config


class FixedNoiseServer:
    def __init__(self, policy, host, port, checkpoint):
        self.policy, self.host, self.port = policy, host, port
        self.meta = dict(policy.metadata)
        self.meta.update({"wam2repair_fixed_noise": True, "policy_config": "pi05_libero", "checkpoint": checkpoint,
                          "checkpoint_marker_sha256": hashlib.sha256((Path(checkpoint)/'TRANSFER_VERIFIED.json').read_bytes()).hexdigest(),
                          "checkpoint_transfer_manifest_sha256": hashlib.sha256((Path(checkpoint)/'transfer_sha256.json').read_bytes()).hexdigest(),
                          "server_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "noise_shape": [10, 32], "policy_action_shape": [10, 7], "protocol": "client must provide internal pi0.5 diffusion noise float32"})

    async def handler(self, websocket):
        packer = msgpack_numpy.Packer(); await websocket.send(packer.pack(self.meta))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                noise = obs.pop("_wam2repair_noise")
                if not isinstance(noise, np.ndarray) or noise.dtype != np.float32:
                    raise ValueError("required noise must be a float32 numpy array")
                if noise.shape != (10, 32) or not np.isfinite(noise).all():
                    raise ValueError(f"invalid required noise: {noise.shape}")
                result = self.policy.infer(obs, noise=noise)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="fixed-noise inference error")
                raise

    async def run(self):
        async def health(connection, request):
            if request.path == "/healthz": return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None
        async with ws_server.serve(self.handler, self.host, self.port, compression=None, max_size=None, process_request=health) as server:
            await server.serve_forever()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, required=True); p.add_argument("--checkpoint", required=True)
    a = p.parse_args(); assert 1 <= a.port <= 65535
    checkpoint=Path(a.checkpoint).resolve(); root=Path('__WAM2REPAIR_ROOT__').resolve()
    assert root in checkpoint.parents and (checkpoint/'TRANSFER_VERIFIED.json').is_file() and (checkpoint/'transfer_sha256.json').is_file()
    # Do not rely on a marker alone: verify every checkpoint file before loading it.
    for record in json.loads((checkpoint/'transfer_sha256.json').read_text()):
        path = checkpoint / record['path']
        assert path.is_file() and path.stat().st_size == record['bytes']
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == record['sha256'], f"checkpoint hash mismatch: {path}"
    policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(checkpoint))
    logging.info("fixed-noise pi05 server host=%s port=%d", socket.gethostname(), a.port)
    asyncio.run(FixedNoiseServer(policy, "127.0.0.1", a.port, str(checkpoint)).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
