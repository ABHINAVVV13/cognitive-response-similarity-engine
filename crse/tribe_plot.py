"""
Save cortical surface figures using Meta TRIBE's ``PlotBrain`` (PyVista backend).

Matches the pattern from TRIBE demos::

    from tribev2.plotting import PlotBrain
    plotter = PlotBrain(mesh=\"fsaverage5\")

Requires optional deps (same idea as ``tribev2[plotting]``)::

    uv pip install -e \".[tribe-plot]\"
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _require_plotting_stack() -> None:
    try:
        import pyvista  # noqa: F401
        from tribev2.plotting import PlotBrain  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TRIBE surface plots need the plotting stack. Install:\n"
            '  uv pip install -e ".[tribe-plot]"'
        ) from e


def _mean_maps(mean_a: np.ndarray, mean_b: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    diff = mean_a.astype(np.float64) - mean_b.astype(np.float64)
    return (
        ("mean_a", mean_a),
        ("mean_b", mean_b),
        ("mean_diff_a_minus_b", diff),
    )


def _render_surface_png_bytes(
    data: np.ndarray,
    plotter,
    *,
    figsize: tuple[float, float] = (6.5, 5.0),
    dpi: int = 110,
    views: str = "left",
    cmap: str = "coolwarm",
    symmetric_cbar: bool = True,
    norm_percentile: float = 98.0,
) -> bytes:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plotter.plot_surf(
        np.asarray(data, dtype=np.float64),
        axes=ax,
        views=views,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        norm_percentile=norm_percentile,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def encode_mean_surface_pngs_base64(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    *,
    mesh: str = "fsaverage5",
    dpi: int = 110,
) -> Dict[str, str]:
    """Render TRIBE surface PNGs in memory; return filename stem → standard base64."""
    _require_plotting_stack()
    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh=mesh)
    out: Dict[str, str] = {}
    for name, data in _mean_maps(mean_a, mean_b):
        raw = _render_surface_png_bytes(data, plotter, dpi=dpi)
        out[name] = base64.standard_b64encode(raw).decode("ascii")
    return out


def save_mean_surface_figures(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    out_dir: str | Path,
    *,
    mesh: str = "fsaverage5",
    views: str = "left",
    cmap: str = "coolwarm",
    symmetric_cbar: bool = True,
    norm_percentile: float = 98.0,
    dpi: int = 150,
) -> List[Path]:
    """Write PNGs for time-mean maps: ``mean_a``, ``mean_b``, ``mean_diff`` (A−B).

    Parameters
    ----------
    mean_a, mean_b
        1D arrays, length = number of fsaverage vertices (LH+RH), TRIBE order.
    """
    _require_plotting_stack()
    from matplotlib import pyplot as plt
    from tribev2.plotting import PlotBrain

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = PlotBrain(mesh=mesh)

    paths: List[Path] = []
    for name, data in _mean_maps(mean_a, mean_b):
        raw = _render_surface_png_bytes(
            data,
            plotter,
            figsize=(8, 6),
            dpi=dpi,
            views=views,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            norm_percentile=norm_percentile,
        )
        path = out_dir / f"{name}.png"
        path.write_bytes(raw)
        paths.append(path)
        logger.info("Wrote surface plot %s", path)

    return paths
