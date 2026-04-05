"""
TRIBE calls `uvx whisperx`, which downloads a full isolated PyTorch/CUDA stack
into ~/.cache/uv at inference time. On RunPod that often hits "No space left on device".

Patch eventstransforms.py to invoke the `whisperx` CLI from the same environment
after we `uv pip install whisperx` in the Docker image.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    path = Path("/app/tribev2/tribev2/eventstransforms.py")
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    # Match: "uvx", newline + indent + "whisperx",
    pattern = r'([ \t]*)"uvx",\s*\n[ \t]*"whisperx",'
    repl = r'\1"whisperx",'
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        print(
            "error: could not find uvx/whisperx subprocess line to patch; "
            "TRIBE upstream may have changed. Inspect ExtractWordsFromAudio.",
            file=sys.stderr,
        )
        sys.exit(1)
    new = new.replace(
        "Running whisperx via uvx...",
        "Running whisperx (image install)...",
        1,
    )
    path.write_text(new, encoding="utf-8")
    print(f"Patched {path} (use whisperx CLI instead of uvx whisperx)")


if __name__ == "__main__":
    main()
