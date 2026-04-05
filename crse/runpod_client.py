"""
RunPod serverless client for CRSE.

Call a deployed CRSE RunPod endpoint from your local machine —
no GPU required locally.  Videos are uploaded to a public URL
(or you provide existing URLs) and processed on the remote GPU worker.

Usage::

    from crse.runpod_client import CRSERunPodClient

    client = CRSERunPodClient(
        api_key="your_runpod_api_key",
        endpoint_id="your_endpoint_id",
    )

    result = client.compare(
        video_a_url="https://example.com/video_a.mp4",
        video_b_url="https://example.com/video_b.mp4",
    )
    print(result)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class CRSERunPodClient:
    """Client to call a CRSE RunPod Serverless endpoint.

    Parameters
    ----------
    api_key : str or None
        RunPod API key. If ``None``, reads from ``RUNPOD_API_KEY`` env var.
    endpoint_id : str or None
        RunPod endpoint ID. If ``None``, reads from ``CRSE_ENDPOINT_ID`` env var.
    timeout : int
        Max seconds to wait for sync requests (default 300 = 5 min).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: int = 300,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("CRSE_ENDPOINT_ID")

        if not self.api_key:
            raise ValueError(
                "RunPod API key required. Pass api_key= or set RUNPOD_API_KEY env var. "
                "Get your key at https://www.runpod.io/console/user/settings"
            )
        if not self.endpoint_id:
            raise ValueError(
                "RunPod endpoint ID required. Pass endpoint_id= or set CRSE_ENDPOINT_ID env var."
            )

        self.timeout = timeout
        self._base_url = f"{RUNPOD_API_BASE}/{self.endpoint_id}"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Sync compare ───────────────────────────────────────────────────

    def compare(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a synchronous comparison via RunPod (blocks until done).

        Parameters
        ----------
        video_a_url, video_b_url : str
            Publicly accessible URLs for the two videos.
        regions : list[str], optional
            Brain regions to analyze. None = all.
        metrics : list[str], optional
            Similarity metrics to compute. None = all.

        Returns
        -------
        dict
            The comparison result as a dictionary (same schema as
            ``ComparisonResult.to_dict()``).

        Raises
        ------
        RuntimeError
            If the job fails or times out.
        """
        payload = self._build_payload(video_a_url, video_b_url, regions, metrics)
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
        elif status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown error')}")
        elif status == "IN_QUEUE" or status == "IN_PROGRESS":
            # runsync timed out — fall back to polling
            job_id = data.get("id")
            logger.info("Job %s still running, falling back to polling...", job_id)
            return self._poll_result(job_id)
        else:
            raise RuntimeError(f"Unexpected status: {status}. Response: {data}")

    # ── Async compare ──────────────────────────────────────────────────

    def compare_async(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Submit an async comparison job. Returns the job ID.

        Use ``get_result(job_id)`` to poll for the result.

        Returns
        -------
        str
            RunPod job ID.
        """
        payload = self._build_payload(video_a_url, video_b_url, regions, metrics)
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
        """Check the status of an async job."""
        resp = requests.get(
            f"{self._base_url}/status/{job_id}",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_result(self, job_id: str, poll_interval: float = 3.0) -> Dict[str, Any]:
        """Poll until the job is done, then return the result."""
        return self._poll_result(job_id, poll_interval)

    # ── Cancel ─────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        resp = requests.post(
            f"{self._base_url}/cancel/{job_id}",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check the health of the endpoint (number of workers, queue depth)."""
        resp = requests.get(
            f"{self._base_url}/health",
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Internal helpers ───────────────────────────────────────────────

    def _build_payload(
        self,
        video_a_url: str,
        video_b_url: str,
        regions: Optional[List[str]],
        metrics: Optional[List[str]],
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
        return payload

    def _poll_result(
        self, job_id: str, poll_interval: float = 3.0
    ) -> Dict[str, Any]:
        """Poll RunPod for job completion."""
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            data = self.get_status(job_id)
            status = data.get("status")

            if status == "COMPLETED":
                logger.info("Job %s completed.", job_id)
                return data.get("output", {})
            elif status == "FAILED":
                raise RuntimeError(
                    f"RunPod job {job_id} failed: {data.get('error', 'unknown')}"
                )
            elif status == "CANCELLED":
                raise RuntimeError(f"RunPod job {job_id} was cancelled.")

            logger.debug("Job %s status: %s — polling again in %.0fs", job_id, status, poll_interval)
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {self.timeout}s. "
            f"Use get_result('{job_id}') to continue polling."
        )
