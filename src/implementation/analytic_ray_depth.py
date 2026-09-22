"""Analytic metric-depth labels from MuJoCo ``mj_ray``.

This deliberately does not instantiate ``mujoco.Renderer``.  It pairs the
robosuite RGB/segmentation product with ray intersections from the *same*
MuJoCo state.  A depth value is usable only when a non-boundary renderer geom
label belongs to the frozen ray geom-group and the ray returns that exact geom
ID.  Thus invalid pixels are explicit rather than silently filled with a
renderer-buffer value.

This is a geometry/raster-alignment utility, not a collision, penetration, or
contact oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco


RAY_GEOMGROUP = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
PIXEL_OFFSET = 0.5


@dataclass(frozen=True)
class DenseRayResult:
    depth_m: np.ndarray
    valid_mask: np.ndarray
    ray_geom_id: np.ndarray
    stable_geom_mask: np.ndarray
    eligible_mask: np.ndarray
    metrics: dict


def _stable_geom_pixels(segmentation: np.ndarray, model) -> np.ndarray:
    """Return non-boundary pixels whose labelled geom is in RAY_GEOMGROUP."""
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 segmentation, got {seg.shape}")
    h, w = seg.shape[:2]
    result = np.zeros((h, w), dtype=bool)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    objtype, geom_id = seg[..., 0], seg[..., 1]
    candidate = (objtype == geom_type) & (geom_id >= 0) & (geom_id < model.ngeom)
    # A 3x3 constant geom-ID neighborhood is a predeclared raster-boundary
    # exclusion. It never depends on ray residuals or depth values.
    for v in range(1, h - 1):
        for u in range(1, w - 1):
            gid = int(geom_id[v, u])
            if candidate[v, u] and model.geom_group[gid] < len(RAY_GEOMGROUP) and RAY_GEOMGROUP[model.geom_group[gid]]:
                result[v, u] = bool(np.all(geom_id[v - 1:v + 2, u - 1:u + 2] == gid))
    return result


def dense_ray_depth(model, data, K, camera_to_world, segmentation, *, wrong_vertical_flip=False) -> DenseRayResult:
    """Generate axial metric depth and an exact-ID validity mask for one view.

    ``segmentation`` must be in the predeclared top-down raster convention used
    by robosuite's camera intrinsics/extrinsics and the frozen B0 ray gate.
    The returned depth has NaN at invalid pixels; consumers must use
    ``valid_mask`` and may not convert invalid pixels into target depth.
    """
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(camera_to_world, dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError(f"invalid camera shapes K={K.shape}, T={T.shape}")
    stable = _stable_geom_pixels(segmentation, model)
    h, w = stable.shape
    ray_ids = np.full((h, w), -1, dtype=np.int32)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    eligible = stable.copy()
    seg_ids = np.asarray(segmentation)[..., 1]
    origin = T[:3, 3].copy()
    for v, u in np.argwhere(eligible):
        ray_v = h - 1 - int(v) if wrong_vertical_flip else int(v)
        vec_cam = np.linalg.solve(K, np.array([float(u) + PIXEL_OFFSET, float(ray_v) + PIXEL_OFFSET, 1.0]))
        vec_cam /= np.linalg.norm(vec_cam)
        vec_world = T[:3, :3] @ vec_cam
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, vec_world, RAY_GEOMGROUP, 1, -1, hit)
        ray_ids[v, u] = hit[0]
        if distance >= 0 and int(hit[0]) == int(seg_ids[v, u]):
            # Never write a finite value for an invalid label. This keeps the
            # stored depth/mask contract true even before a caller's gate.
            depth[v, u] = np.float32(distance * vec_cam[2])
            valid[v, u] = True
    eligible_count = int(eligible.sum())
    hit_mask = eligible & (ray_ids >= 0)
    hit_count = int(hit_mask.sum())
    matched_count = int((eligible & (ray_ids == seg_ids)).sum())
    valid_count = int(valid.sum())
    metrics = {
        "pixel_offset": PIXEL_OFFSET,
        "axial_depth": True,
        "boundary_policy": "3x3 constant renderer geom ID; selected MuJoCo geomgroup only",
        "eligible_pixel_count": eligible_count,
        "ray_hit_count": hit_count,
        "ray_geom_id_agreement_fraction": matched_count / max(1, eligible_count),
        "valid_pixel_count": valid_count,
        "valid_coverage_fraction": valid_count / max(1, eligible_count),
        "invalid_depth_is_nan": bool(np.isnan(depth[~valid]).all()),
    }
    return DenseRayResult(depth, valid, ray_ids, stable, eligible, metrics)


def require_exact_alignment(result: DenseRayResult, *, minimum_pixels: int = 100) -> None:
    """Fail closed for a factual label source; no residual-based relaxation."""
    m = result.metrics
    if (m["eligible_pixel_count"] < minimum_pixels or
            m["ray_hit_count"] != m["eligible_pixel_count"] or
            m["ray_geom_id_agreement_fraction"] != 1.0 or
            m["valid_coverage_fraction"] != 1.0 or
            not m["invalid_depth_is_nan"]):
        raise ValueError("analytic ray/raster alignment failed: " + str(m))
