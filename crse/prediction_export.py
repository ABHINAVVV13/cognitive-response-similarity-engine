"""
Export TRIBE prediction arrays for visualization (Three.js, nilearn, etc.).

TRIBE outputs shape ``(n_timesteps, n_vertices)`` on fsaverage5 (~20k vertices).
Similarity-only JSON is small; full tensors are not. This module builds optional
payloads:

* **summary** — time-mean activation per vertex for A, B, and A−B (compact JSON).
* **npz_b64** — gzip-style compressed numpy blobs (base64), for full ``(T,V)``
  reconstruction in Python; can be large for RunPod responses.

Mesh geometry (sphere/inflated coordinates, faces) is **not** included: use
standard fsaverage5 surfaces (e.g. ``nilearn.datasets.fetch_surf_fsaverage()``)
and align vertex index ``i`` with ``mean_a[i]``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Literal, Optional

import numpy as np

ExportMode = Literal["none", "summary", "npz_b64"]


def build_visualization_payload(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    mode: ExportMode = "summary",
    *,
    max_timesteps_npz: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable dict for ``mode`` (skip if ``none``)."""
    if mode == "none":
        return {}

    if preds_a.shape[1] != preds_b.shape[1]:
        raise ValueError(
            f"Vertex mismatch: A {preds_a.shape} vs B {preds_b.shape}"
        )

    ta, tb = preds_a.shape[0], preds_b.shape[0]
    if ta != tb:
        t_common = min(ta, tb)
        a = preds_a[:t_common].astype(np.float32, copy=False)
        b = preds_b[:t_common].astype(np.float32, copy=False)
    else:
        a = preds_a.astype(np.float32, copy=False)
        b = preds_b.astype(np.float32, copy=False)

    out: Dict[str, Any] = {
        "schema": "crse/predictions/v1",
        "n_vertices": int(a.shape[1]),
        "n_timesteps_used": int(a.shape[0]),
        "mesh": "fsaverage5",
        "note": (
            "Vertex i matches fsaverage5 surface order (use nilearn or FreeSurfer "
            "fsaverage5 LH+RH combined ordering as in TRIBE)."
        ),
    }

    if mode == "summary":
        mean_a = np.nanmean(a, axis=0)
        mean_b = np.nanmean(b, axis=0)
        diff = mean_a - mean_b
        # JSON-friendly finite values
        out["mean_activation_a"] = _finite_list(mean_a)
        out["mean_activation_b"] = _finite_list(mean_b)
        out["mean_diff_a_minus_b"] = _finite_list(diff)
        return out

    if mode == "npz_b64":
        a_w = a
        b_w = b
        if max_timesteps_npz is not None and a.shape[0] > max_timesteps_npz:
            a_w = a[:max_timesteps_npz]
            b_w = b[:max_timesteps_npz]
            out["npz_timesteps_truncated_to"] = max_timesteps_npz
        out["prediction_a_npz_b64"] = _npz_b64(a_w, "pred")
        out["prediction_b_npz_b64"] = _npz_b64(b_w, "pred")
        out["shape_a"] = list(a_w.shape)
        out["shape_b"] = list(b_w.shape)
        out["decode_hint"] = (
            "Python: import base64, io, numpy as np; "
            "d=np.load(io.BytesIO(base64.standard_b64decode(b64))); "
            "arr=d['pred']"
        )
        return out

    raise ValueError(f"Unknown export mode: {mode!r}")


def _finite_list(arr: np.ndarray) -> list[float]:
    x = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(v) for v in x.tolist()]


def _npz_b64(arr: np.ndarray, key: str = "pred") -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{key: arr.astype(np.float32, copy=False)})
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")
