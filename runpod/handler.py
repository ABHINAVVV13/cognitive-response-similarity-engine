"""
RunPod Serverless Handler for CRSE.

This handler runs on a GPU worker. It accepts two video URLs,
downloads them, runs TRIBE v2 brain encoding, computes similarity,
and returns the full ComparisonResult as JSON.

Deploy as a RunPod serverless endpoint via Docker.

Input schema (job["input"])::

    {
        "video_a_url": "https://example.com/video_a.mp4",
        "video_b_url": "https://example.com/video_b.mp4",
        "regions": ["emotional_limbic", "visual_cortex"],  // optional
        "metrics": ["cosine_similarity", "pearson_correlation"]  // optional
    }

Output::

    {
        "video_a": "/tmp/video_a.mp4",
        "video_b": "/tmp/video_b.mp4",
        "whole_brain": { ... },
        "regions": [ ... ],
        ...
    }
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crse.runpod")

# ───────────────────────────────────────────────────────────────────────────
# Global model loading — happens ONCE during cold start, not per-request
# ───────────────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("CRSE_MODEL_ID", "facebook/tribev2")
CACHE_DIR = os.environ.get("CRSE_CACHE_DIR", "/cache")
DEVICE = os.environ.get("CRSE_DEVICE", "auto")

logger.info("Initializing CRSE engine (model=%s, device=%s)...", MODEL_ID, DEVICE)
from crse.engine import CRSEngine  # noqa: E402

ENGINE = CRSEngine(
    model_id=MODEL_ID,
    cache_folder=CACHE_DIR,
    device=DEVICE,
)
# Force model load during cold start so first request is fast
ENGINE._ensure_model()
logger.info("CRSE engine ready.")


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def download_video(url: str, dest_dir: str) -> str:
    """Download a video from a URL to a local file. Returns local path."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "video.mp4"
    # Ensure valid video extension
    if not any(filename.endswith(ext) for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm")):
        filename += ".mp4"
    local_path = os.path.join(dest_dir, filename)

    logger.info("Downloading %s → %s", url, local_path)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    file_size = os.path.getsize(local_path)
    logger.info("Downloaded %s (%.1f MB)", filename, file_size / 1e6)
    return local_path


def validate_input(job_input: dict) -> tuple[str, str, list | None, list | None]:
    """Validate and extract input parameters."""
    video_a_url = job_input.get("video_a_url")
    video_b_url = job_input.get("video_b_url")

    if not video_a_url or not video_b_url:
        raise ValueError(
            "Both 'video_a_url' and 'video_b_url' are required in the input. "
            f"Got: video_a_url={video_a_url!r}, video_b_url={video_b_url!r}"
        )

    regions = job_input.get("regions", None)
    metrics = job_input.get("metrics", None)

    if regions is not None and not isinstance(regions, list):
        raise ValueError(f"'regions' must be a list of strings, got {type(regions)}")
    if metrics is not None and not isinstance(metrics, list):
        raise ValueError(f"'metrics' must be a list of strings, got {type(metrics)}")

    return video_a_url, video_b_url, regions, metrics


# ───────────────────────────────────────────────────────────────────────────
# RunPod handler
# ───────────────────────────────────────────────────────────────────────────


def handler(job: dict) -> dict:
    """
    Process a CRSE comparison job.

    Downloads two videos from URLs, runs TRIBE v2 predictions,
    computes similarity metrics, returns results as JSON dict.
    """
    job_input = job.get("input", {})
    t0 = time.time()

    try:
        video_a_url, video_b_url, regions, metrics = validate_input(job_input)
    except ValueError as e:
        return {"error": str(e)}

    # Override engine regions if specified
    if regions:
        ENGINE._regions = regions
        ENGINE._brain_mgr = __import__("crse.brain_regions", fromlist=["BrainRegionManager"]).BrainRegionManager()

    # Download videos to temp directory
    with tempfile.TemporaryDirectory(prefix="crse_") as tmp_dir:
        try:
            # Use unique names to avoid collision
            video_a_path = download_video(video_a_url, tmp_dir)
            # Ensure video B has a different name if URLs have same filename
            video_b_dest = os.path.join(tmp_dir, "video_b")
            os.makedirs(video_b_dest, exist_ok=True)
            video_b_path = download_video(video_b_url, video_b_dest)
        except requests.RequestException as e:
            return {"error": f"Failed to download video: {str(e)}"}

        try:
            result = ENGINE.compare(
                video_a=video_a_path,
                video_b=video_b_path,
                metrics=metrics,
            )
        except Exception as e:
            logger.exception("CRSE comparison failed")
            return {"error": f"Comparison failed: {str(e)}"}

    # Convert to dict for JSON serialization
    output = result.to_dict()
    output["total_time_seconds"] = round(time.time() - t0, 2)
    output["worker_device"] = DEVICE

    logger.info(
        "Job complete in %.1fs — whole-brain cosine: %.4f",
        output["total_time_seconds"],
        output["whole_brain"].get("cosine_similarity", float("nan")),
    )
    return output


# ───────────────────────────────────────────────────────────────────────────
# Start the RunPod serverless worker
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
