"""
RunPod serverless client for CRSE.

Workers always return cortical PNGs as ``surface_pngs_base64``. Pass
``render_brain=True`` for full (T,V) predictions in ``visualization`` (large).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


def save_surface_pngs_from_result(result: Dict[str, Any], out_dir: str | Path) -> List[Path]:
    """Decode ``surface_pngs_base64`` into ``mean_*.png`` files (skips ``_error`` keys)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b64_map = result.get("surface_pngs_base64")
    if not isinstance(b64_map, dict):
        return []
    written: List[Path] = []
    for stem, blob in b64_map.items():
        if not isinstance(stem, str) or stem.startswith("_"):
            continue
        if not isinstance(blob, str):
            continue
        path = out / f"{stem}.png"
        path.write_bytes(base64.standard_b64decode(blob))
        written.append(path)
    return written


class CRSERunPodClient:
    """Client for a CRSE RunPod Serverless endpoint."""

    DEFAULT_TIMEOUT = 1800

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("CRSE_ENDPOINT_ID")

        if not self.api_key:
            raise ValueError(
                "RunPod API key required. Pass api_key= or set RUNPOD_API_KEY. "
                "https://www.runpod.io/console/user/settings"
            )
        if not self.endpoint_id:
            raise ValueError(
                "RunPod endpoint ID required. Pass endpoint_id= or set CRSE_ENDPOINT_ID."
            )

        if timeout is not None:
            self.timeout = timeout
        else:
            env_t = os.environ.get("CRSE_RUNPOD_TIMEOUT")
            self.timeout = int(env_t) if env_t else self.DEFAULT_TIMEOUT
        self._base_url = f"{RUNPOD_API_BASE}/{self.endpoint_id}"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def compare(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        render_brain: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous compare. Blocks until the job completes."""
        payload = self._build_payload(
            video_a_url, video_b_url, regions, metrics, render_brain=render_brain
        )
        logger.info("Submitting sync job to RunPod endpoint %s ...", self.endpoint_id)

        resp = requests.post(
            f"{self._base_url}/runsync",
            headers=self._headers,
            json=payload,
            timeout=self.timeout + 10,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        if status == "COMPLETED":
            logger.info("Job completed successfully.")
            return data.get("output", {})
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown error')}")
        if status in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = data.get("id")
            logger.info("Job %s still running, falling back to polling...", job_id)
            return self._poll_result(job_id)
        raise RuntimeError(f"Unexpected status: {status}. Response: {data}")

    def compare_async(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        render_brain: bool = False,
    ) -> str:
        """Submit async job; returns job id."""
        payload = self._build_payload(
            video_a_url, video_b_url, regions, metrics, render_brain=render_brain
        )
        logger.info("Submitting async job to RunPod endpoint %s ...", self.endpoint_id)

        resp = requests.post(
            f"{self._base_url}/run",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        logger.info("Async job submitted: %s", job_id)
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        resp = requests.get(
            f"{self._base_url}/status/{job_id}",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_result(self, job_id: str, poll_interval: float = 3.0) -> Dict[str, Any]:
        return self._poll_result(job_id, poll_interval)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self._base_url}/cancel/{job_id}",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = requests.get(
            f"{self._base_url}/health",
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _build_payload(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]],
        metrics: Optional[List[str]],
        *,
        render_brain: bool = False,
    ) -> dict:
        payload: dict = {
            "input": {
                "video_a_url": video_a_url,
                "video_b_url": video_b_url,
            }
        }
        if regions:
            payload["input"]["regions"] = regions
        if metrics:
            payload["input"]["metrics"] = metrics
        if render_brain:
            payload["input"]["render_brain"] = True
        return payload

    def _poll_result(
        self, job_id: str, poll_interval: float = 3.0
    ) -> Dict[str, Any]:
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            data = self.get_status(job_id)
            status = data.get("status")

            if status == "COMPLETED":
                logger.info("Job %s completed.", job_id)
                return data.get("output", {})
            if status == "FAILED":
                raise RuntimeError(
                    f"RunPod job {job_id} failed: {data.get('error', 'unknown')}"
                )
            if status == "CANCELLED":
                raise RuntimeError(f"RunPod job {job_id} was cancelled.")

            logger.debug(
                "Job %s status: %s — polling again in %.0fs",
                job_id,
                status,
                poll_interval,
            )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {self.timeout}s. "
            f"Set CRSE_RUNPOD_TIMEOUT or crse compare --runpod --runpod-timeout SECONDS."
        )
