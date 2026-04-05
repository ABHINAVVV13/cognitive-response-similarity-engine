"""
Core engine for the Cognitive Response Similarity Engine.

``CRSEngine`` wraps Meta TRIBE v2 to:

1. Accept two video paths
2. Predict cortical fMRI-like brain responses for each
3. Compute per-region and whole-brain similarity metrics
4. Package the results into a rich ``ComparisonResult``
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from crse.brain_regions import BrainRegionManager
from crse.similarity import ALL_METRICS, compute_all_metrics

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Result container
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class RegionScore:
    """Similarity scores for a single brain region."""

    name: str
    description: str
    n_vertices: int
    metrics: Dict[str, float]

    @property
    def mean_score(self) -> float:
        """Average of all metric values (quick overall indicator)."""
        vals = [v for v in self.metrics.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0


@dataclass
class ComparisonResult:
    """Full result of comparing two videos through CRSE.

    Attributes
    ----------
    video_a : str
        Path to the first video.
    video_b : str
        Path to the second video.
    whole_brain : Dict[str, float]
        Whole-brain similarity metrics.
    regions : List[RegionScore]
        Per-region breakdowns.
    prediction_shape_a : tuple
        Shape of the predicted activation for video A ``(T, V)``.
    prediction_shape_b : tuple
        Shape of the predicted activation for video B ``(T, V)``.
    elapsed_seconds : float
        Total wall-clock time for the comparison.
    """

    video_a: str
    video_b: str
    whole_brain: Dict[str, float]
    regions: List[RegionScore]
    prediction_shape_a: tuple
    prediction_shape_b: tuple
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Human-readable summary ─────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║       COGNITIVE RESPONSE SIMILARITY ENGINE — RESULTS       ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Video A : {self.video_a}",
            f"  Video B : {self.video_b}",
            f"  Pred. A : {self.prediction_shape_a}  (timesteps × vertices)",
            f"  Pred. B : {self.prediction_shape_b}  (timesteps × vertices)",
            f"  Time    : {self.elapsed_seconds:.1f}s",
            "",
            "── Whole-Brain Similarity ──────────────────────────────────────",
        ]
        for metric, score in self.whole_brain.items():
            bar = _score_bar(score)
            lines.append(f"  {metric:<30s}  {score:+.4f}  {bar}")

        lines.append("")
        lines.append("── Per-Region Breakdown ────────────────────────────────────────")
        for region in self.regions:
            lines.append(f"\n  ▸ {region.name}  ({region.n_vertices:,} vertices)")
            for metric, score in region.metrics.items():
                bar = _score_bar(score)
                lines.append(f"      {metric:<28s}  {score:+.4f}  {bar}")

        lines.append("")
        lines.append("════════════════════════════════════════════════════════════════")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "video_a": self.video_a,
            "video_b": self.video_b,
            "whole_brain": self.whole_brain,
            "regions": [
                {
                    "name": r.name,
                    "description": r.description,
                    "n_vertices": r.n_vertices,
                    "metrics": r.metrics,
                    "mean_score": r.mean_score,
                }
                for r in self.regions
            ],
            "prediction_shape_a": list(self.prediction_shape_a),
            "prediction_shape_b": list(self.prediction_shape_b),
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Write results JSON to a file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")
        logger.info("Results saved to %s", path)


# ───────────────────────────────────────────────────────────────────────────
# Main engine
# ───────────────────────────────────────────────────────────────────────────


class CRSEngine:
    """Cognitive Response Similarity Engine.

    Loads Meta TRIBE v2 once and exposes a ``compare()`` method that
    accepts two video paths and returns a :class:`ComparisonResult`.

    Parameters
    ----------
    model_id : str
        HuggingFace repo id or local path for the TRIBE v2 checkpoint.
    cache_folder : str
        Directory for caching extracted features.
    device : str
        PyTorch device string (``"auto"`` selects CUDA if available).
    regions : list[str] | None
        Subset of region names to analyse.  ``None`` = all defined regions.
    """

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        cache_folder: str = "./cache",
        device: str = "auto",
        regions: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.cache_folder = cache_folder
        self.device = device
        self._regions = regions
        self._model = None
        self._brain_mgr = BrainRegionManager()

    # ── Lazy model loading ─────────────────────────────────────────────

    def _ensure_model(self):
        """Load TRIBE v2 on first use."""
        if self._model is not None:
            return
        logger.info("Loading TRIBE v2 from %s ...", self.model_id)
        from tribev2 import TribeModel

        self._model = TribeModel.from_pretrained(
            self.model_id,
            cache_folder=self.cache_folder,
            device=self.device,
        )
        logger.info("TRIBE v2 model loaded successfully.")

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, video_path: str | Path) -> np.ndarray:
        """Predict brain responses for a single video.

        Parameters
        ----------
        video_path : str or Path
            Path to a video file (.mp4, .avi, .mkv, .mov, .webm).

        Returns
        -------
        np.ndarray
            Predicted activations of shape ``(n_timesteps, n_vertices)``.
        """
        self._ensure_model()
        video_path = str(Path(video_path).resolve())
        logger.info("Predicting brain response for: %s", video_path)
        events = self._model.get_events_dataframe(video_path=video_path)
        preds, _segments = self._model.predict(events=events)
        logger.info("  → prediction shape: %s", preds.shape)
        return preds

    # ── Comparison ─────────────────────────────────────────────────────

    def compare(
        self,
        video_a: str | Path,
        video_b: str | Path,
        metrics: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """Compare two videos and return a :class:`ComparisonResult`.

        Parameters
        ----------
        video_a, video_b : str or Path
            Paths to the two video files to compare.
        metrics : list[str] or None
            Subset of metric names to compute.  ``None`` = all metrics.

        Returns
        -------
        ComparisonResult
        """
        t0 = time.time()

        # 1. Predict
        preds_a = self.predict(video_a)
        preds_b = self.predict(video_b)

        # 2. Select metrics
        metric_fns = ALL_METRICS
        if metrics:
            metric_fns = {k: v for k, v in ALL_METRICS.items() if k in metrics}

        # 3. Whole-brain similarity
        whole_brain = {name: fn(preds_a, preds_b) for name, fn in metric_fns.items()}

        # 4. Per-region similarity
        region_names = self._regions or self._brain_mgr.get_region_names()
        region_scores: list[RegionScore] = []

        for rname in region_names:
            try:
                mask = self._brain_mgr.get_region_mask(rname)
                # Ensure mask aligns with prediction vertex count
                n_verts = preds_a.shape[1]
                if len(mask) > n_verts:
                    mask = mask[:n_verts]
                elif len(mask) < n_verts:
                    mask = np.pad(mask, (0, n_verts - len(mask)), constant_values=False)

                n_roi = int(mask.sum())
                if n_roi == 0:
                    logger.warning("Region '%s' has 0 matched vertices — skipping.", rname)
                    continue

                a_roi = preds_a[:, mask]
                b_roi = preds_b[:, mask]

                roi_metrics = {name: fn(a_roi, b_roi) for name, fn in metric_fns.items()}
                region_scores.append(
                    RegionScore(
                        name=rname,
                        description=self._brain_mgr.get_region_description(rname),
                        n_vertices=n_roi,
                        metrics=roi_metrics,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to compute region '%s': %s", rname, exc)

        elapsed = time.time() - t0

        return ComparisonResult(
            video_a=str(video_a),
            video_b=str(video_b),
            whole_brain=whole_brain,
            regions=region_scores,
            prediction_shape_a=preds_a.shape,
            prediction_shape_b=preds_b.shape,
            elapsed_seconds=round(elapsed, 2),
            metadata={
                "model_id": self.model_id,
                "device": self.device,
                "n_metrics": len(metric_fns),
                "n_regions": len(region_scores),
            },
        )


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _score_bar(score: float, width: int = 20) -> str:
    """Render a small Unicode bar chart for a [-1, 1] score."""
    if not np.isfinite(score):
        return "  [  n/a  ]"
    # Normalise to [0, 1]
    norm = (score + 1.0) / 2.0
    norm = max(0.0, min(1.0, norm))
    filled = int(round(norm * width))
    return "▏" + "█" * filled + "░" * (width - filled) + "▏"
