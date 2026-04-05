"""
Downloader utility for fetching web videos and YouTube links.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    try:
        result = urlparse(path)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def download_video(url: str, dest_dir: str) -> str:
    """Download a video from a URL to a local file. Returns local path."""
    parsed = urlparse(url)
    
    # ── YouTube URL Handling ──────────────────────────────
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        logger.info("Detected YouTube URL: %s", url)
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("yt-dlp is not installed. Please install it to use YouTube links.")
        
        video_uid = str(uuid.uuid4())[:8]
        out_tmpl = os.path.join(dest_dir, f"youtube_{video_uid}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'merge_output_format': 'mp4',
            'outtmpl': out_tmpl,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Fetching YouTube media (capped at 720p) via yt-dlp...")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
        # Due to merge_output_format, yt-dlp usually produces .mp4 regardless of raw extension
        base_path = os.path.splitext(filename)[0]
        final_path = base_path + ".mp4"
        if not os.path.exists(final_path) and os.path.exists(filename):
            final_path = filename
            
        file_size = os.path.getsize(final_path)
        logger.info("Downloaded YouTube video (%.1f MB) to %s", file_size / 1e6, final_path)
        return final_path

    # ── Standard Raw File Download ────────────────────────
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
