"""
Brain region definitions on the fsaverage5 cortical mesh.

Uses the Destrieux atlas (2009) via ``nilearn`` to carve the ~20,484-vertex
fsaverage5 surface into functionally meaningful regions of interest (ROIs)
relevant to emotional processing, visual perception, auditory processing,
and language comprehension.

The atlas labels are cached on first use so the lookup is near-instant
for subsequent calls.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Destrieux atlas label groups
# Each group maps a human-readable region name to a list of substrings
# that match label names in the Destrieux atlas.  Matching is case-
# insensitive and uses "contains" logic.
# ───────────────────────────────────────────────────────────────────────────

REGION_LABEL_PATTERNS: dict[str, list[str]] = {
    # ── Emotional / Limbic ─────────────────────────────────────────────
    "emotional_limbic": [
        "insula",                        # insular cortex — interoception, emotion
        "cingul",                        # cingulate cortex (anterior & posterior)
        "Pole_temporal",                 # temporal pole — social/emotional processing
        "Front_Inf_Orb",                 # orbitofrontal cortex — reward, valence
        "orbital",                       # orbital sulci
        "subcallosal",                   # subcallosal area
        "subcentral",                    # peri-insular
    ],

    # ── Visual Cortex ──────────────────────────────────────────────────
    "visual_cortex": [
        "calcarine",                     # V1 — primary visual cortex
        "cuneus",                        # V2/V3
        "lingual",                       # ventral visual stream
        "Occipital",                     # lateral occipital complex
        "occipito-temporal",             # fusiform / ventral object recognition
        "Pole_occipital",                # occipital pole
        "parieto_occipital",             # dorsal visual stream
        "collateral_transverse",         # collateral sulcus (scene processing)
    ],

    # ── Auditory Cortex ────────────────────────────────────────────────
    "auditory_cortex": [
        "Heschl",                        # primary auditory cortex (A1)
        "Transv_temporal",               # transverse temporal gyrus
        "planum_temporale",              # auditory association
        "Lat_Fissure",                   # Sylvian fissure border
    ],

    # ── Language Network ───────────────────────────────────────────────
    "language_network": [
        "Front_Inf_Tri",                 # Broca's area — pars triangularis
        "Front_Inf_Opercular",           # Broca's area — pars opercularis
        "Sup_Temporal",                  # superior temporal gyrus (Wernicke's)
        "angular",                       # angular gyrus — semantic integration
        "supramarginal",                 # supramarginal gyrus — phonological
        "Temporal_Sup",                  # STS — speech perception
    ],

    # ── Motor / Somatosensory ──────────────────────────────────────────
    "motor_somatosensory": [
        "Precentral",                    # primary motor cortex (M1)
        "Postcentral",                   # primary somatosensory cortex (S1)
        "paracentral",                   # supplementary motor area
    ],

    # ── Prefrontal / Executive ─────────────────────────────────────────
    "prefrontal_executive": [
        "Front_Sup",                     # dorsolateral PFC
        "Front_Middle",                  # mid-DLPFC
        "Front_Inf",                     # IFG (overlaps language — kept separate)
        "frontopolar",                   # frontopolar cortex — planning
        "rostral",                       # medial PFC
        "straight",                      # gyrus rectus
    ],

    # ── Default Mode Network (DMN) ─────────────────────────────────────
    "default_mode_network": [
        "precuneus",                     # precuneus — self-referential
        "Parietal_Inf",                  # inferior parietal lobule
        "Front_Med",                     # medial frontal
        "Temp_Mid",                      # middle temporal gyrus
        "temporoparietal",               # TPJ — theory of mind
    ],
}


@dataclass
class BrainRegionManager:
    """Load and cache fsaverage5 Destrieux atlas labels and provide ROI masks.

    Attributes
    ----------
    mesh : str
        Freesurfer mesh name (default ``"fsaverage5"``).
    n_vertices_per_hemi : int
        Number of vertices per hemisphere (10,242 for fsaverage5).
    """

    mesh: str = "fsaverage5"
    n_vertices_per_hemi: int = 10_242
    _labels_lh: np.ndarray | None = field(default=None, repr=False)
    _labels_rh: np.ndarray | None = field(default=None, repr=False)
    _label_names: list[str] = field(default_factory=list, repr=False)

    # ── Atlas loading ──────────────────────────────────────────────────

    @functools.cached_property
    def _atlas(self):
        """Fetch the Destrieux atlas via nilearn (downloaded once, cached)."""
        from nilearn import datasets

        data_dir = os.environ.get("NILEARN_DATA", "").strip()
        if data_dir:
            data_dir = os.path.expanduser(data_dir)
            return datasets.fetch_atlas_surf_destrieux(data_dir=data_dir)
        return datasets.fetch_atlas_surf_destrieux()

    def _ensure_loaded(self) -> None:
        if self._labels_lh is None:
            atlas = self._atlas
            self._labels_lh = np.asarray(atlas["map_left"])
            self._labels_rh = np.asarray(atlas["map_right"])
            self._label_names = [
                label.decode() if isinstance(label, bytes) else str(label)
                for label in atlas["labels"]
            ]
            logger.info(
                "Loaded Destrieux atlas: %d labels, %d + %d vertices",
                len(self._label_names),
                len(self._labels_lh),
                len(self._labels_rh),
            )

    # ── Label inspection ───────────────────────────────────────────────

    @property
    def label_names(self) -> list[str]:
        """All atlas label names."""
        self._ensure_loaded()
        return list(self._label_names)

    @property
    def n_vertices(self) -> int:
        """Total number of vertices (both hemispheres)."""
        return self.n_vertices_per_hemi * 2

    # ── ROI masks ──────────────────────────────────────────────────────

    def _match_label_indices(self, patterns: list[str]) -> list[int]:
        """Return atlas label indices whose name contains any of *patterns*."""
        self._ensure_loaded()
        indices = []
        for idx, name in enumerate(self._label_names):
            name_lower = name.lower()
            if any(p.lower() in name_lower for p in patterns):
                indices.append(idx)
        return indices

    def get_region_mask(self, region_name: str) -> np.ndarray:
        """Boolean mask (length ``n_vertices``) for the named ROI.

        Parameters
        ----------
        region_name : str
            One of the keys in ``REGION_LABEL_PATTERNS`` (e.g.
            ``"emotional_limbic"``, ``"visual_cortex"``).

        Returns
        -------
        np.ndarray[bool]
            Boolean array of length ``n_vertices``.

        Raises
        ------
        KeyError
            If *region_name* is not a defined group.
        """
        if region_name not in REGION_LABEL_PATTERNS:
            raise KeyError(
                f"Unknown region '{region_name}'.  "
                f"Available: {list(REGION_LABEL_PATTERNS.keys())}"
            )
        self._ensure_loaded()
        patterns = REGION_LABEL_PATTERNS[region_name]
        matched_ids = self._match_label_indices(patterns)

        mask_lh = np.isin(self._labels_lh, matched_ids)
        mask_rh = np.isin(self._labels_rh, matched_ids)

        # Trim/pad to expected vertex count (atlas may have slightly
        # different length than model output)
        mask = np.concatenate([
            self._pad_mask(mask_lh, self.n_vertices_per_hemi),
            self._pad_mask(mask_rh, self.n_vertices_per_hemi),
        ])
        return mask

    def get_all_region_masks(self) -> Dict[str, np.ndarray]:
        """Return masks for every defined region."""
        return {name: self.get_region_mask(name) for name in REGION_LABEL_PATTERNS}

    def get_region_names(self) -> List[str]:
        """List of available region names."""
        return list(REGION_LABEL_PATTERNS.keys())

    def get_region_description(self, region_name: str) -> str:
        """Human-readable description of a brain region group."""
        descriptions = {
            "emotional_limbic": (
                "Emotional & Limbic regions: insula, cingulate cortex, temporal pole, "
                "orbitofrontal cortex — involved in emotion processing, interoception, "
                "and affective valuation."
            ),
            "visual_cortex": (
                "Visual Cortex: calcarine sulcus (V1), cuneus, lingual gyrus, lateral "
                "occipital complex, fusiform — responsible for visual perception and "
                "object recognition."
            ),
            "auditory_cortex": (
                "Auditory Cortex: Heschl's gyrus (A1), planum temporale, transverse "
                "temporal — primary and associative auditory processing."
            ),
            "language_network": (
                "Language Network: Broca's area (IFG), Wernicke's area (STG/STS), "
                "angular gyrus, supramarginal gyrus — speech production, comprehension, "
                "and semantic integration."
            ),
            "motor_somatosensory": (
                "Motor & Somatosensory: precentral (M1) and postcentral (S1) gyri, "
                "paracentral lobule — voluntary movement and tactile perception."
            ),
            "prefrontal_executive": (
                "Prefrontal / Executive: dorsolateral PFC, medial PFC, frontopolar "
                "cortex — working memory, planning, cognitive control."
            ),
            "default_mode_network": (
                "Default Mode Network: precuneus, inferior parietal, medial frontal, "
                "middle temporal, TPJ — self-referential thought, mind-wandering, "
                "theory of mind."
            ),
        }
        return descriptions.get(region_name, f"Region: {region_name}")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _pad_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
        """Trim or zero-pad *mask* to *target_len*."""
        if len(mask) >= target_len:
            return mask[:target_len]
        return np.pad(mask, (0, target_len - len(mask)), constant_values=False)
