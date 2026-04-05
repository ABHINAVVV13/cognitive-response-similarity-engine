"""
Similarity metrics for comparing predicted brain activation patterns.

All functions accept two arrays of shape ``(n_timesteps, n_vertices)`` and
return a scalar similarity score in [-1, 1] or [0, 1] depending on the metric.
Region-masked versions are obtained by slicing columns before calling these.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Spatial similarity  (collapse time → compare spatial patterns)
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between mean activation vectors.

    Averages each array over time to produce a single spatial pattern
    vector per stimulus, then computes the cosine of the angle between them.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)
        Predicted brain activations for each video.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  1 = identical patterns.
    """
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    denom = np.linalg.norm(a_mean) * np.linalg.norm(b_mean)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_mean, b_mean) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between mean spatial activation patterns.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)

    Returns
    -------
    float
        Pearson *r* in [-1, 1].
    """
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    r, _ = stats.pearsonr(a_mean, b_mean)
    return float(r)


def spatial_pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Voxel-wise spatial pattern correlation (mean-centred).

    Each vertex's activation is mean-centred before computing Pearson r,
    emphasising relative *patterns* rather than overall magnitude.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)

    Returns
    -------
    float
        Pearson *r* of the demeaned spatial patterns.
    """
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_centred = a_mean - a_mean.mean()
    b_centred = b_mean - b_mean.mean()
    denom = np.linalg.norm(a_centred) * np.linalg.norm(b_centred)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_centred, b_centred) / denom)


# ---------------------------------------------------------------------------
# Temporal similarity  (compare how activations evolve over time)
# ---------------------------------------------------------------------------


def temporal_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Average per-vertex temporal correlation between activation time-courses.

    For each vertex, computes the Pearson correlation of its time-course
    across the two stimuli (after aligning lengths).  Returns the mean
    across all vertices.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)
        The arrays are trimmed to the shorter length along T.

    Returns
    -------
    float
        Mean Pearson *r* across vertices.
    """
    min_t = min(a.shape[0], b.shape[0])
    a = a[:min_t]
    b = b[:min_t]
    if min_t < 3:
        # Not enough timepoints for meaningful correlation
        return cosine_similarity(a, b)

    correlations = []
    for v in range(a.shape[1]):
        r, _ = stats.pearsonr(a[:, v], b[:, v])
        if np.isfinite(r):
            correlations.append(r)
    return float(np.mean(correlations)) if correlations else 0.0


def temporal_isc(a: np.ndarray, b: np.ndarray) -> float:
    """Inter-stimulus correlation (ISC) inspired metric.

    Computes the correlation of the *spatial mean* time-courses — i.e.
    the global signal similarity.  Useful as a quick, noise-robust measure.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)

    Returns
    -------
    float
        Pearson *r* of spatially-averaged time-courses.
    """
    min_t = min(a.shape[0], b.shape[0])
    ts_a = a[:min_t].mean(axis=1)
    ts_b = b[:min_t].mean(axis=1)
    if min_t < 3:
        return 0.0
    r, _ = stats.pearsonr(ts_a, ts_b)
    return float(r) if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Representational similarity analysis (RSA)
# ---------------------------------------------------------------------------


def representational_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Representational Similarity Analysis between two activation matrices.

    Builds a temporal representational dissimilarity matrix (RDM) for each
    stimulus — correlations between all pairs of time-points — then
    correlates the two RDMs (Spearman).  A high value means the two videos
    create similar *structure* in representational space.

    Parameters
    ----------
    a, b : np.ndarray, shape (T, V)

    Returns
    -------
    float
        Spearman *r* between the upper-triangular RDM entries.
    """
    min_t = min(a.shape[0], b.shape[0])
    a = a[:min_t]
    b = b[:min_t]
    if min_t < 4:
        return cosine_similarity(a, b)

    rdm_a = squareform(pdist(a, metric="correlation"))
    rdm_b = squareform(pdist(b, metric="correlation"))
    upper = np.triu_indices_from(rdm_a, k=1)
    r, _ = stats.spearmanr(rdm_a[upper], rdm_b[upper])
    return float(r) if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

ALL_METRICS = {
    "cosine_similarity": cosine_similarity,
    "pearson_correlation": pearson_correlation,
    "spatial_pattern": spatial_pattern_similarity,
    "temporal_correlation": temporal_correlation,
    "temporal_isc": temporal_isc,
    "representational_similarity": representational_similarity,
}


def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Run every registered similarity metric and return a name → score dict."""
    return {name: fn(a, b) for name, fn in ALL_METRICS.items()}
