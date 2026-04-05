"""
Visualization utilities for CRSE comparison results.

Provides brain surface plots, radar charts, and temporal similarity
time-series using ``matplotlib`` and ``nilearn``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Radar / spider chart of per-region similarity
# ───────────────────────────────────────────────────────────────────────────


def plot_similarity_radar(
    result,
    metric: str = "pearson_correlation",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Radar chart showing a single metric across all brain regions.

    Parameters
    ----------
    result : ComparisonResult
    metric : str
        Which metric to plot on each axis.
    title : str, optional
    save_path : str or Path, optional
        If given, save the figure to this path.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    regions = result.regions
    if not regions:
        logger.warning("No region data to plot.")
        return plt.figure()

    labels = [r.name.replace("_", "\n") for r in regions]
    values = [r.metrics.get(metric, 0.0) for r in regions]

    # Close the polygon
    values += values[:1]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Plot
    ax.plot(angles, values, "o-", linewidth=2, color="#58a6ff", markersize=6)
    ax.fill(angles, values, alpha=0.2, color="#58a6ff")

    # Axis styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color="#c9d1d9", fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(["-0.5", "0", "0.5", "1.0"], color="#8b949e", size=8)
    ax.yaxis.grid(True, color="#30363d", linewidth=0.5)
    ax.xaxis.grid(True, color="#30363d", linewidth=0.5)
    ax.spines["polar"].set_color("#30363d")

    if title is None:
        title = f"Neural Response Similarity — {metric}"
    ax.set_title(title, color="#f0f6fc", size=13, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Radar chart saved to %s", save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────────
# Multi-metric bar chart
# ───────────────────────────────────────────────────────────────────────────


def plot_metric_bars(
    result,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Grouped bar chart: one group per region, one bar per metric.

    Parameters
    ----------
    result : ComparisonResult
    """
    regions = result.regions
    if not regions:
        return plt.figure()

    metric_names = list(regions[0].metrics.keys())
    n_metrics = len(metric_names)
    n_regions = len(regions)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(n_regions)
    width = 0.8 / n_metrics
    colors = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#79c0ff", "#ffa657"]

    for i, metric in enumerate(metric_names):
        vals = [r.metrics.get(metric, 0.0) for r in regions]
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            label=metric.replace("_", " "),
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r.name.replace("_", "\n") for r in regions],
        color="#c9d1d9", size=8, fontweight="bold",
    )
    ax.set_ylabel("Similarity Score", color="#c9d1d9", size=10)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right", fontsize=7,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
    )

    if title is None:
        title = "CRSE — Per-Region Similarity Breakdown"
    ax.set_title(title, color="#f0f6fc", size=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Bar chart saved to %s", save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────────
# Whole-brain similarity summary
# ───────────────────────────────────────────────────────────────────────────


def plot_whole_brain_summary(
    result,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Horizontal bar chart of whole-brain metrics."""
    metrics = result.whole_brain
    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#7ee787" if v >= 0 else "#f85149" for v in values]
    ax.barh(names, values, color=colors, alpha=0.85, edgecolor="none", height=0.6)

    ax.set_xlim(-1, 1)
    ax.axvline(0, color="#30363d", linewidth=0.8)
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.set_xlabel("Similarity Score", color="#c9d1d9", size=10)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "CRSE — Whole-Brain Similarity",
        color="#f0f6fc", size=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Summary chart saved to %s", save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────────
# Brain surface map (using nilearn)
# ───────────────────────────────────────────────────────────────────────────


def plot_brain_surface(
    activation: np.ndarray,
    title: str = "Predicted Brain Activation",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot mean activation on the fsaverage5 cortical surface.

    Parameters
    ----------
    activation : np.ndarray, shape (T, V) or (V,)
        Predicted activations.  If 2-D, averaged over time.
    """
    try:
        from nilearn import datasets, plotting
    except ImportError:
        logger.warning("nilearn not available — skipping brain surface plot.")
        return plt.figure()

    if activation.ndim == 2:
        activation = activation.mean(axis=0)

    n_verts = len(activation)
    half = n_verts // 2
    lh_data = activation[:half]
    rh_data = activation[half:]

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("#0d1117")

    for ax, hemi, data, mesh_key in [
        (axes[0], "Left", lh_data, "infl_left"),
        (axes[1], "Right", rh_data, "infl_right"),
    ]:
        plotting.plot_surf_stat_map(
            fsaverage[mesh_key],
            stat_map=data,
            hemi=hemi.lower(),
            view="lateral",
            colorbar=True,
            cmap="coolwarm",
            title=f"{title} — {hemi} Hemisphere",
            axes=ax,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Brain surface plot saved to %s", save_path)
    return fig
