"""One-context pi0.5 downstream probe using raw/repaired WAM frames.

This is an explicit imagined-frame interface, not native pi0.5 conditioning:
after a fixed, factual 16-action prefix, only the image portion of the next
pi0.5 query is replaced by an imagined frame. Measured robot state remains
real. The simulator future is never read for action selection.
"""
import argparse
import hashlib
import json
import traceback
from copy import copy
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation



ROOT = Path("__WAM2REPAIR_ROOT__").resolve()


class FixedNoiseClient:
    def __init__(self, port):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.packer = msgpack_numpy.Packer(); self.unpack = msgpack_numpy.unpackb
        self.ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, ping_interval=None, open_timeout=60)
        self.metadata = self.unpack(self.ws.recv(timeout=60))
        checkpoint = ROOT / "models/openpi/official/pi05_libero"
        assert self.metadata.get("wam2repair_fixed_noise") is True
        assert self.metadata.get("policy_config") == "pi05_libero" and self.metadata.get("noise_shape") == [10, 32]
        assert self.metadata.get("policy_action_shape") == [10, 7]
        assert self.metadata.get("checkpoint") == str(checkpoint)
        assert self.metadata.get("checkpoint_marker_sha256") == digest(checkpoint / "TRANSFER_VERIFIED.json")
        assert self.metadata.get("checkpoint_transfer_manifest_sha256") == digest(checkpoint / "transfer_sha256.json")
        assert self.metadata.get("server_code_sha256") == digest(Path(__file__).with_name("pi05_fixed_noise_server.py"))

    def infer(self, obs, prompt, noise):
        from openpi_client import image_tools
        assert isinstance(noise, np.ndarray) and noise.dtype == np.float32
        assert noise.shape == (10, 32) and np.isfinite(noise).all()
        payload = {"observation/image": image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224),
                   "observation/wrist_image": image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224),
                   "observation/state": np.concatenate((obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec(), obs["robot0_gripper_qpos"])).astype(np.float32),
                   "prompt": prompt, "_wam2repair_noise": noise}
        self.ws.send(self.packer.pack(payload)); raw = self.ws.recv(timeout=1200)
        if isinstance(raw, str): raise RuntimeError(raw)
        actions = np.asarray(self.unpack(raw)["actions"], dtype=np.float32)
        assert actions.shape == (10, 7) and np.isfinite(actions).all()
        return actions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def policy_to_raw(frame, resolution=256):
    """Turn policy-oriented HWC image into raw LIBERO convention for Client.infer."""
    image = np.asarray(Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((resolution, resolution), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(image[::-1, ::-1])


def canonical_dual(frame):
    """Exact 2x downsample used by F.interpolate bilinear align_corners=False here."""
    x=np.asarray(frame,dtype=np.uint8)
    assert x.ndim==3 and x.shape[2]==3 and x.shape[1]==2*x.shape[0]
    if x.shape == (224,448,3):
        # At integer scale 1/2, bilinear samples lie at 2x2 cell centers.
        x=x.astype(np.float32).reshape(112,2,224,2,3).mean(axis=(1,3))
        return np.rint(x).clip(0,255).astype(np.uint8)
    assert x.shape == (112,224,3)
    return x


def synthetic_observation(obs, dual_frame):
    image = np.asarray(dual_frame, dtype=np.uint8)
    assert image.ndim == 3 and image.shape[2] == 3 and image.shape[1] % 2 == 0
    half = image.shape[1] // 2
    result = copy(obs)
    result["agentview_image"] = policy_to_raw(image[:, :half])
    result["robot0_eye_in_hand_image"] = policy_to_raw(image[:, half:])
    return result


def reset_to_prefix(env, init, seed, prefix, expected_qpos, expected_qvel, expected_time, expected_images):
    env.seed(seed); env.reset(); obs = env.set_init_state(init)
    for _ in range(10):
        obs, *_ = env.step([0.] * 6 + [-1.])
    for action in prefix:
        obs, *_ = env.step(action.tolist())
    err = float(np.max(np.abs(np.asarray(env.sim.data.qpos) - expected_qpos)))
    vel_err = float(np.max(np.abs(np.asarray(env.sim.data.qvel) - expected_qvel)))
    time_err = abs(float(env.sim.data.time) - float(expected_time))
    if max(err, vel_err, time_err) > 1e-6:
        raise RuntimeError(f"prefix replay mismatch: qpos={err}, qvel={vel_err}, time={time_err}")
    image_errors = {}
    for archived_key, obs_key in (("agentview_rgb", "agentview_image"), ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand_image")):
        expected, actual = expected_images[archived_key], np.asarray(obs[obs_key])
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        image_errors[archived_key] = {"observation_key": obs_key, "exact": bool(np.array_equal(actual, expected)), "max_abs": int(delta.max())}
        if not image_errors[archived_key]["exact"]:
            raise RuntimeError(f"prefix replay RGB mismatch for {archived_key}: max_abs={image_errors[archived_key]['max_abs']}")
    return obs, {"qpos": err, "qvel": vel_err, "sim_time": time_err, "images": image_errors}


def run_condition(env, init, seed, prefix, expected_qpos, expected_qvel, expected_time, expected_images, client, instruction, condition, imagined, noise_bank, max_steps):
    obs, reset_error = reset_to_prefix(env, init, seed, prefix, expected_qpos, expected_qvel, expected_time, expected_images)
    policy_obs = obs if condition == "actual" else synthetic_observation(obs, imagined)
    first_chunk = client.infer(policy_obs, instruction, noise_bank[0])
    actions, query_sources = [], [condition]
    success = bool(env.check_success())
    for action in first_chunk[:5]:
        obs, _, _, _ = env.step(action.tolist()); actions.append(action.copy()); success = bool(env.check_success())
        if success: break
    # Subsequent decisions receive actual observations in every condition.
    while len(actions) < max_steps and not success:
        assert len(query_sources) < len(noise_bank), "noise bank exhausted"
        chunk = client.infer(obs, instruction, noise_bank[len(query_sources)]); query_sources.append("actual")
        for action in chunk[:5]:
            obs, _, _, _ = env.step(action.tolist()); actions.append(action.copy()); success = bool(env.check_success())
            if success or len(actions) >= max_steps: break
    return {"success": success, "post_prefix_steps": len(actions), "first_chunk": first_chunk.tolist(),
            "executed_actions": np.asarray(actions), "query_sources": query_sources, "prefix_replay_errors": reset_error,
            "noise_sha256": [hashlib.sha256(noise_bank[i].tobytes()).hexdigest() for i in range(len(query_sources))]}


def run(a):
    trajectory_npz = a.trajectory / "trajectory.npz"
    with np.load(trajectory_npz, allow_pickle=False) as d:
        prefix = d["actions"][:16].copy(); expected_qpos = d["qpos"][16].copy(); expected_qvel=d["qvel"][16].copy(); expected_time=d["sim_time"][16].copy()
        expected_images = {key: d[key][16].copy() for key in ("agentview_rgb", "robot0_eye_in_hand_rgb")}
        assert not bool(d['success'][16]), 'prefix already succeeds; invalid downstream intervention context'
    meta = json.loads((a.trajectory / "result.json").read_text())
    checkpoint = ROOT / "models/openpi/official/pi05_libero"
    assert meta["suite"] == "libero_spatial" and meta["task"] == 0 and meta["episode"] == 6
    assert meta["action_mode"] == "policy" and Path(meta["checkpoint"]).resolve() == checkpoint and meta["status"] == "EXECUTED"
    assert prefix.shape == (16, 7)
    bank = json.loads((a.bank / "manifest.json").read_text()); wam_result = json.loads((a.wam_result / "result.json").read_text())
    repair_meta = json.loads(a.repair_meta.read_text()); repair_result = json.loads(a.repair_result.read_text())
    raw_path = a.raw_wam / "window_0000_wam.npz"
    record = bank["records"][0]
    assert bank["source_sha256"] == digest(trajectory_npz) and len(bank["records"]) == 1
    assert bank["horizon"] == 16 and bank["frame_offsets"] == [0, 4, 8, 12, 16] and Path(bank["policy_checkpoint"]).resolve() == checkpoint
    assert record["id"] == "window_0000" and record["group"] == "libero_spatial/task0/episode6" and record["split"] == "test" and record["start"] == 0
    input_path, target_path = a.bank / record["input"], a.bank / record["target"]
    assert digest(input_path) == record["input_sha256"] and digest(target_path) == record["target_sha256"]
    with np.load(input_path, allow_pickle=False) as input_data:
        assert set(input_data.files) == {"current_rgb", "proprio", "actions"}
        np.testing.assert_array_equal(input_data["actions"], prefix)
    assert wam_result["status"] == "GENERATED_ALIGNED" and wam_result["manifest_sha256"] == digest(a.bank / "manifest.json")
    assert len(wam_result["records"]) == 1 and wam_result["records"][0]["sha256"] == digest(raw_path)
    assert repair_meta["runtime_file"] == a.repair.name and repair_meta["runtime_sha256"] == digest(a.repair)
    assert repair_meta["repair_result_sha256"] == digest(a.repair_result)
    assert repair_meta["test_wam_result_sha256"] == digest(a.wam_result / "result.json")
    assert repair_meta["source_predictions_sha256"] == repair_result["heldout_prediction"]["sha256"]
    assert repair_meta["test_groups"] == ["libero_spatial/task0/episode6"] and repair_result["hashes"]["test_wam"] == digest(a.wam_result / "result.json")
    server_launch = json.loads(a.server_launch.read_text())
    assert server_launch["status"] == "RUNNING_FIXED_NOISE_SERVER" and server_launch["port"] == a.port
    assert server_launch["checkpoint"] == str(checkpoint) and server_launch["server_code_sha256"] == digest(Path(__file__).with_name("pi05_fixed_noise_server.py"))
    with np.load(raw_path, allow_pickle=False) as d:
        raw = d["frames"][-1].copy()
    with np.load(a.repair, allow_pickle=False) as d:
        assert set(d.files) == {"repair"}
        repaired = d["repair"][-1].copy()
    assert raw.dtype == np.uint8 and raw.shape == (224, 448, 3)
    assert repaired.dtype.kind == "f" and np.isfinite(repaired).all() and repaired.min() >= 0 and repaired.max() <= 1
    assert repaired.ndim == 3 and repaired.shape[1] == 2 * repaired.shape[0]
    repaired = np.rint(repaired * 255).clip(0, 255).astype(np.uint8)
    raw, repaired = canonical_dual(raw), canonical_dual(repaired)
    noise_bank = np.random.default_rng(np.random.SeedSequence([meta["seed"], 20260918])).standard_normal((64, 10, 32), dtype=np.float32)
    np.savez_compressed(a.out / "noise_bank.npz", noise=noise_bank)
    noise_bank_sha256 = digest(a.out / "noise_bank.npz")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[meta["suite"]](); task = suite.get_task(meta["task"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    client = FixedNoiseClient(a.port); outcomes = {}
    try:
        # Repeated identical query checks server/model determinism before any environment action.
        check_env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=-1)
        try:
            check_obs, _ = reset_to_prefix(check_env, suite.get_task_init_states(meta["task"])[meta["episode"]], meta["seed"], prefix, expected_qpos, expected_qvel, expected_time, expected_images)
            repeat_a, repeat_b = client.infer(check_obs, task.language, noise_bank[0]), client.infer(check_obs, task.language, noise_bank[0])
            np.testing.assert_array_equal(repeat_a, repeat_b)
        finally:
            check_env.close()
        for name, imagined in (("actual", None), ("raw_wam", raw), ("repaired_wam", repaired)):
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=-1)
            try:
                outcomes[name] = run_condition(env, suite.get_task_init_states(meta["task"])[meta["episode"]], meta["seed"], prefix,
                                               expected_qpos, expected_qvel, expected_time, expected_images, client, task.language, name, imagined, noise_bank, a.max_post_prefix_steps)
            finally:
                env.close()
        actual = np.asarray(outcomes["actual"]["first_chunk"], dtype=np.float32)
        for name in ("raw_wam", "repaired_wam"):
            outcomes[name]["first_chunk_l2_vs_actual"] = float(np.linalg.norm(np.asarray(outcomes[name]["first_chunk"], dtype=np.float32) - actual))
        actions_path = a.out / "actions.npz"
        np.savez_compressed(actions_path, **{k: v["executed_actions"] for k, v in outcomes.items()})
        for value in outcomes.values(): value.pop("executed_actions")
        report = {"status": "COMPLETED_IMAGINED_FRAME_PROBE", "scope": "one paired pi0.5 context; imagined-frame injection is not native policy input and does not establish general VLA gain",
                  "conditions": outcomes,
                  "inputs_sha256": {"trajectory": digest(trajectory_npz), "trajectory_result": digest(a.trajectory / "result.json"), "bank_manifest": digest(a.bank / "manifest.json"), "bank_input": digest(input_path), "bank_target": digest(target_path), "wam_result": digest(a.wam_result / "result.json"), "raw_wam": digest(raw_path), "runtime_repair": digest(a.repair), "runtime_repair_meta": digest(a.repair_meta), "repair_result": digest(a.repair_result), "server_launch": digest(a.server_launch)},
                  "artifacts_sha256": {"noise_bank": noise_bank_sha256, "actions": digest(actions_path)},
                  "code_sha256": digest(Path(__file__)), "no_runtime_factual_future": True,
                  "fixed_noise": {"seed": meta["seed"], "shape": [10, 32], "policy_action_shape": [10, 7], "noise_bank_sha256": noise_bank_sha256, "noise_bank_entries": len(noise_bank), "repeat_identical_action": True, "server_metadata": client.metadata},
                  "image_protocol": {"raw_wam": "224x448 uint8 -> exact 2x2 mean/round -> 112x224", "repaired_wam": "already 112x224 float[0,1] -> uint8", "both": "each 112x112 camera is PIL-bilinear resized to 256 then client resize/pad to 224 and reversed into LIBERO raw convention"},
                  "limitations": ["one context", "previously evaluated and inspected episode6; post-hoc diagnostic, not sealed or confirmatory", "no candidate ranking", "no physical correction certification", "repaired frame is OOD for pi0.5"]}
        atomic_json(a.out / "result.json", report)
        print(json.dumps(report), flush=True)
    finally:
        client.ws.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-post-prefix-steps", type=int, default=204)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--raw-wam", type=Path, required=True)
    p.add_argument("--repair", type=Path, required=True)
    p.add_argument("--repair-meta", type=Path, required=True)
    p.add_argument("--repair-result", type=Path, required=True)
    p.add_argument("--wam-result", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--server-launch", type=Path, required=True)
    a = p.parse_args(); assert a.max_post_prefix_steps >= 5
    for path in (a.out, a.trajectory, a.raw_wam, a.repair, a.repair_meta, a.repair_result, a.wam_result, a.bank, a.server_launch):
        assert ROOT == path.resolve() or ROOT in path.resolve().parents, f"personal path required: {path}"
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except Exception:
        failure = {"status": "FAILED_IMAGINED_FRAME_PROBE", "code_sha256": digest(Path(__file__)),
                   "traceback": traceback.format_exc()}
        atomic_json(a.out / "result.json", failure)
        raise


if __name__ == "__main__": main()
