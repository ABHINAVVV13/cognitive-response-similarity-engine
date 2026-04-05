"""
RunPod Serverless Handler for CRSE.

Input (job["input"])::

    {
        "video_a_url": "https://...",
        "video_b_url": "https://...",
        "regions": [...],       // optional
        "metrics": [...],       // optional
        "render_brain": false  // optional; large npz_b64 + same as CLI --render-brain
    }

Every job returns similarity scores plus ``surface_pngs_base64`` (cortical PNGs).
When ``render_brain`` is true, ``visualization`` contains full (T,V) predictions
(base64 npz) — very large.
"""

import logging
import os
import tempfile
import time

import requests
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crse.runpod")

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
ENGINE._ensure_model()
logger.info("CRSE engine ready.")

from crse.downloader import download_video  # noqa: E402


def validate_input(
    job_input: dict,
) -> tuple[str, str, list | None, list | None, bool]:
    video_a_url = job_input.get("video_a_url")
    video_b_url = job_input.get("video_b_url")

    if not video_a_url or not video_b_url:
        raise ValueError(
            "Both 'video_a_url' and 'video_b_url' are required in the input. "
            f"Got: video_a_url={video_a_url!r}, video_b_url={video_b_url!r}"
        )

    regions = job_input.get("regions", None)
    metrics = job_input.get("metrics", None)
    render_brain = job_input.get("render_brain", False)

    if regions is not None and not isinstance(regions, list):
        raise ValueError(f"'regions' must be a list of strings, got {type(regions)}")
    if metrics is not None and not isinstance(metrics, list):
        raise ValueError(f"'metrics' must be a list of strings, got {type(metrics)}")
    if not isinstance(render_brain, bool):
        raise ValueError(f"'render_brain' must be a boolean, got {type(render_brain).__name__}")

    return (video_a_url, video_b_url, regions, metrics, render_brain)


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    t0 = time.time()

    try:
        video_a_url, video_b_url, regions, metrics, render_brain = validate_input(job_input)
    except ValueError as e:
        return {"error": str(e)}

    if regions:
        ENGINE._regions = regions
        ENGINE._brain_mgr = __import__(
            "crse.brain_regions", fromlist=["BrainRegionManager"]
        ).BrainRegionManager()

    with tempfile.TemporaryDirectory(prefix="crse_") as tmp_dir:
        try:
            video_a_path = download_video(video_a_url, tmp_dir)
            video_b_dest = os.path.join(tmp_dir, "video_b")
            os.makedirs(video_b_dest, exist_ok=True)
            video_b_path = download_video(video_b_url, video_b_dest)
        except requests.RequestException as e:
            return {"error": f"Failed to download video: {str(e)}"}

        try:
            result = ENGINE.compare(
                video_a_path,
                video_b_path,
                metrics=metrics,
                out_dir=None,
                render_brain=False,
                return_surface_pngs_base64=True,
                return_predictions_npz_b64=render_brain,
            )
        except Exception as e:
            logger.exception("CRSE comparison failed")
            return {"error": f"Comparison failed: {str(e)}"}

    output = result.to_dict()
    output["total_time_seconds"] = round(time.time() - t0, 2)
    output["worker_device"] = DEVICE

    logger.info(
        "Job complete in %.1fs — whole-brain cosine: %.4f",
        output["total_time_seconds"],
        output["whole_brain"].get("cosine_similarity", float("nan")),
    )
    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
