"""
Cognitive Response Similarity Engine (CRSE)
===========================================

Compare neural response patterns between videos using Meta TRIBE v2.

Given two videos, CRSE predicts cortical brain activation maps using the
TRIBE v2 multimodal brain-encoding model, then computes similarity across
functionally meaningful brain regions — visual cortex, auditory cortex,
language networks, and emotional/limbic regions.

Quick start::

    from crse import CRSEngine

    engine = CRSEngine()
    result = engine.compare("a.mp4", "b.mp4", out_dir="crse_out")  # + brain/*.png
    print(result.summary())
"""

__version__ = "0.1.0"

from crse.engine import CRSEngine, ComparisonResult  # noqa: F401
