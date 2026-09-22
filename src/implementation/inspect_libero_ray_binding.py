"""Read-only environment probe for choosing a ray API compatible with LIBERO's sim binding."""
import json
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(4)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                         camera_depths=True, render_gpu_device_id=-1)
try:
    sim = env.sim
    print(json.dumps({
        "sim_type": str(type(sim)), "model_type": str(type(sim.model)), "data_type": str(type(sim.data)),
        "raw_model_type": str(type(sim.model._model)), "raw_data_type": str(type(sim.data._data)),
        "sim_attrs": [x for x in dir(sim) if not x.startswith("__")],
        "model_attrs": [x for x in dir(sim.model) if "ptr" in x.lower() or "model" in x.lower() or "raw" in x.lower()],
        "data_attrs": [x for x in dir(sim.data) if "ptr" in x.lower() or "data" in x.lower() or "raw" in x.lower()],
        "sim_ray_names": [x for x in dir(sim) if "ray" in x.lower()],
        "model_ray_names": [x for x in dir(sim.model) if "ray" in x.lower()],
        "data_ray_names": [x for x in dir(sim.data) if "ray" in x.lower()],
    }, indent=2))
finally:
    env.close()
