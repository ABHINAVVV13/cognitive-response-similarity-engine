"""
Export an interactive WebGL brain viewer (Three.js) + raw prediction binaries.

Open the viewer over HTTP (browsers block file:// fetches)::

    cd crse_out/viewer && python -m http.server 8765
    # visit http://localhost:8765/

Mesh uses fsaverage5 inflated surfaces (LH then RH), matching typical TRIBE
vertex ordering (``n_vertices`` must match ``preds.shape[1]``).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

GEO_MAGIC = b"CRSEGEO1"


def _load_fsaverage5_combined_mesh() -> tuple[np.ndarray, np.ndarray]:
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    coord_l, faces_l = load_surf_mesh(fs.infl_left)
    coord_r, faces_r = load_surf_mesh(fs.infl_right)
    n_l = int(coord_l.shape[0])
    verts = np.vstack([coord_l.astype(np.float32), coord_r.astype(np.float32)])
    faces = np.vstack([faces_l.astype(np.int32), faces_r.astype(np.int32) + n_l])
    return verts, faces


def write_geometry_bin(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write ``vertices`` (N,3) float32 and ``faces`` (F,3) int32."""
    v = np.ascontiguousarray(vertices.astype(np.float32))
    f = np.ascontiguousarray(faces.astype(np.int32))
    n_verts, n_faces = v.shape[0], f.shape[0]
    with path.open("wb") as fp:
        fp.write(GEO_MAGIC)
        fp.write(struct.pack("<II", n_verts, n_faces))
        fp.write(v.tobytes())
        fp.write(f.tobytes())


def export_interactive_viewer(
    viewer_dir: str | Path,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> None:
    """Write ``geometry.bin``, ``series_a.bin``, ``series_b.bin``, ``meta.json``, ``index.html``."""
    viewer_dir = Path(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    a = np.ascontiguousarray(np.nan_to_num(preds_a.astype(np.float32), nan=0.0))
    b = np.ascontiguousarray(np.nan_to_num(preds_b.astype(np.float32), nan=0.0))
    ta, va = a.shape
    tb, vb = b.shape
    if va != vb:
        raise ValueError(f"Vertex mismatch: A {a.shape} vs B {b.shape}")

    verts, faces = _load_fsaverage5_combined_mesh()
    if verts.shape[0] != va:
        raise ValueError(
            f"Mesh / prediction vertex count mismatch: mesh has {verts.shape[0]} "
            f"vertices but predictions have {va}. TRIBE must use fsaverage5 LH+RH order."
        )

    write_geometry_bin(viewer_dir / "geometry.bin", verts, faces)
    (viewer_dir / "series_a.bin").write_bytes(a.tobytes())
    (viewer_dir / "series_b.bin").write_bytes(b.tobytes())

    meta: Dict[str, Any] = {
        "schema": "crse/brain_viewer/v1",
        "mesh": "fsaverage5_inflated_lh_rh",
        "V": va,
        "T_a": ta,
        "T_b": tb,
    }
    (viewer_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (viewer_dir / "index.html").write_text(_VIEWER_HTML, encoding="utf-8")


def load_predictions_from_visualization_dict(viz: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray] | None:
    """Decode ``prediction_*_npz_b64`` from a RunPod ``visualization`` object."""
    import base64
    import io

    if not isinstance(viz, dict):
        return None
    ba = viz.get("prediction_a_npz_b64")
    bb = viz.get("prediction_b_npz_b64")
    if not isinstance(ba, str) or not isinstance(bb, str):
        return None
    a = np.load(io.BytesIO(base64.standard_b64decode(ba)))["pred"]
    b = np.load(io.BytesIO(base64.standard_b64decode(bb)))["pred"]
    return np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)


_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CRSE — brain viewer</title>
  <style>
    body { margin:0; overflow:hidden; background:#0d1117; color:#c9d1d9; font-family:system-ui,sans-serif; }
    #bar {
      position:absolute; top:0; left:0; right:0; z-index:10;
      display:flex; flex-wrap:wrap; gap:12px; align-items:center;
      padding:10px 14px; background:#161b22; border-bottom:1px solid #30363d;
    }
    #bar label { font-size:13px; display:flex; align-items:center; gap:8px; }
    #t { width:min(420px, 50vw); }
    canvas { display:block; }
  </style>
</head>
<body>
<div id="bar">
  <label>Surface <select id="which"><option value="a">Video A</option><option value="b">Video B</option></select></label>
  <label>Frame <input type="range" id="t" min="0" value="0"/></label>
  <span id="info" style="font-size:13px;opacity:0.9"></span>
  <span style="font-size:12px;opacity:0.55;margin-left:auto">Serve this folder: <code>python -m http.server 8765</code></span>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
const MAGIC = new TextEncoder().encode("CRSEGEO1");

async function loadBinary(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.arrayBuffer();
}

function coolwarm(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {
    const u = t * 2;
    return new THREE.Color().setRGB(0.23 * (1-u) + u, 0.30 * (1-u) + 0.7*u, 0.71 * (1-u) + 0.9*u);
  }
  const u = (t - 0.5) * 2;
  return new THREE.Color().setRGB(0.7 * (1-u) + 0.86*u, 0.7 * (1-u) + 0.08*u, 0.9 * (1-u) + 0.24*u);
}

function colorForValue(v, lo, hi) {
  const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-9);
  return coolwarm((v / m + 1) * 0.5);
}

(async function main() {
  const meta = await (await fetch("meta.json")).json();
  const V = meta.V | 0, Ta = meta.T_a | 0, Tb = meta.T_b | 0;

  const [geoBuf, bufA, bufB] = await Promise.all([
    loadBinary("geometry.bin"),
    loadBinary("series_a.bin"),
    loadBinary("series_b.bin"),
  ]);

  const geo = new DataView(geoBuf);
  for (let i = 0; i < 8; i++) {
    if (geo.getUint8(i) !== MAGIC[i]) throw new Error("Bad geometry.bin header");
  }
  const nVerts = geo.getUint32(8, true);
  const nFaces = geo.getUint32(12, true);
  const vStart = 16;
  const fStart = vStart + nVerts * 3 * 4;
  const pos = new Float32Array(geoBuf, vStart, nVerts * 3);
  const idx = new Uint32Array(geoBuf, fStart, nFaces * 3);

  const flatA = new Float32Array(bufA);
  const flatB = new Float32Array(bufB);
  if (flatA.length !== Ta * V || flatB.length !== Tb * V) {
    throw new Error("series length mismatch");
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 5000);
  camera.position.set(0, 0, 350);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setIndex(new THREE.Uint32BufferAttribute(idx, 1));
  const colors = new Float32Array(nVerts * 3);
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);

  const whichEl = document.getElementById("which");
  const tEl = document.getElementById("t");
  const info = document.getElementById("info");

  function currentTmax() { return whichEl.value === "a" ? Ta : Tb; }
  function rowFlat(which, t) {
    const T = which === "a" ? Ta : Tb;
    const flat = which === "a" ? flatA : flatB;
    const off = t * V;
    return { flat, off, T };
  }

  function paintFrame() {
    const w = whichEl.value;
    const Tm = currentTmax();
    let t = parseInt(tEl.value, 10) | 0;
    if (t >= Tm) t = Tm - 1;
    if (t < 0) t = 0;
    tEl.max = String(Math.max(0, Tm - 1));

    const { flat, off } = rowFlat(w, t);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < V; i++) {
      const v = flat[off + i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const cattr = geom.attributes.color;
    for (let i = 0; i < V; i++) {
      const col = colorForValue(flat[off + i], lo, hi);
      cattr.array[i * 3] = col.r;
      cattr.array[i * 3 + 1] = col.g;
      cattr.array[i * 3 + 2] = col.b;
    }
    cattr.needsUpdate = true;
    info.textContent = "Frame " + t + " / " + (Tm - 1) + "  ·  " + (w === "a" ? "A" : "B");
  }

  tEl.addEventListener("input", paintFrame);
  whichEl.addEventListener("change", () => { tEl.value = "0"; paintFrame(); });

  let drag = false, px = 0, py = 0;
  renderer.domElement.addEventListener("mousedown", e => { drag = true; px = e.clientX; py = e.clientY; });
  window.addEventListener("mouseup", () => { drag = false; });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    mesh.rotation.y += (e.clientX - px) * 0.005;
    mesh.rotation.x += (e.clientY - py) * 0.005;
    px = e.clientX; py = e.clientY;
  });

  window.addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  paintFrame();

  function tick() {
    requestAnimationFrame(tick);
    renderer.render(scene, camera);
  }
  tick();
})().catch(err => {
  document.body.innerHTML = "<pre style='padding:20px;color:#f85149'>" + err + "</pre>";
});
</script>
</body>
</html>
"""
