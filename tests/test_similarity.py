"""
Unit tests for the CRSE similarity metrics module.

Uses synthetic data with known statistical properties to verify
that each metric returns the expected values.
"""

from __future__ import annotations

import numpy as np
import pytest

from crse.similarity import (
    ALL_METRICS,
    compute_all_metrics,
    cosine_similarity,
    pearson_correlation,
    representational_similarity,
    spatial_pattern_similarity,
    temporal_correlation,
    temporal_isc,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def identical_data():
    """Two identical activation matrices."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 100))
    return data, data.copy()


@pytest.fixture
def orthogonal_data():
    """Two uncorrelated activation matrices."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((20, 100))
    b = rng.standard_normal((50, 100))
    return a, b


@pytest.fixture
def anticorrelated_data():
    """Two perfectly negatively correlated matrices."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((20, 100))
    return a, -a


@pytest.fixture
def short_data():
    """Very short time-series (edge case)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, 50)), rng.standard_normal((2, 50))


# ── Tests ──────────────────────────────────────────────────────────────


class TestCosine:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_anticorrelated(self, anticorrelated_data):
        a, b = anticorrelated_data
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_range(self, orthogonal_data):
        a, b = orthogonal_data
        score = cosine_similarity(a, b)
        assert -1.0 <= score <= 1.0

    def test_zero_input(self):
        a = np.zeros((10, 50))
        b = np.ones((10, 50))
        assert cosine_similarity(a, b) == 0.0


class TestPearson:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert pearson_correlation(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_anticorrelated(self, anticorrelated_data):
        a, b = anticorrelated_data
        assert pearson_correlation(a, b) == pytest.approx(-1.0, abs=1e-6)


class TestSpatialPattern:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert spatial_pattern_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_anticorrelated(self, anticorrelated_data):
        a, b = anticorrelated_data
        assert spatial_pattern_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


class TestTemporalCorrelation:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert temporal_correlation(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_short_data_fallback(self, short_data):
        """With <3 timepoints, should fall back to cosine."""
        a, b = short_data
        score = temporal_correlation(a, b)
        assert -1.0 <= score <= 1.0

    def test_different_lengths(self):
        """Should handle different-length inputs by truncation."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((30, 50))
        b = rng.standard_normal((20, 50))
        score = temporal_correlation(a, b)
        assert -1.0 <= score <= 1.0


class TestTemporalISC:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert temporal_isc(a, b) == pytest.approx(1.0, abs=1e-6)


class TestRSA:
    def test_identical(self, identical_data):
        a, b = identical_data
        assert representational_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_short_fallback(self, short_data):
        a, b = short_data
        score = representational_similarity(a, b)
        assert -1.0 <= score <= 1.0


class TestComputeAll:
    def test_returns_all_metrics(self, identical_data):
        a, b = identical_data
        results = compute_all_metrics(a, b)
        assert set(results.keys()) == set(ALL_METRICS.keys())
        for name, val in results.items():
            assert isinstance(val, float), f"{name} should be float"
            assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9 or not np.isfinite(val)

    def test_identical_all_high(self, identical_data):
        a, b = identical_data
        results = compute_all_metrics(a, b)
        for name, val in results.items():
            assert val > 0.9, f"{name} should be ~1.0 for identical data, got {val}"
