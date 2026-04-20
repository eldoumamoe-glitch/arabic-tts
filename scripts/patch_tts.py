"""
Patch Coqui TTS for PyTorch 2.11+ compatibility.

PyTorch 2.11 changed torch.load() default from weights_only=False to True.
XTTS-v2 checkpoints contain config objects that require weights_only=False.

Usage:
    conda activate new-arabic-tts
    python scripts/patch_tts.py
"""

import site
import os
import sys


def patch():
    site_packages = site.getsitepackages()[0]
    io_path = os.path.join(site_packages, "TTS", "utils", "io.py")

    if not os.path.exists(io_path):
        print(f"ERROR: {io_path} not found. Is Coqui TTS installed?")
        sys.exit(1)

    with open(io_path, "r") as f:
        content = f.read()

    if "weights_only=False" in content:
        print("Already patched.")
        return

    patched = content.replace(
        "return torch.load(f, map_location=map_location, **kwargs)",
        "return torch.load(f, map_location=map_location, weights_only=False, **kwargs)",
    )

    count = content.count("return torch.load(f, map_location=map_location, **kwargs)")
    if count == 0:
        print("ERROR: Could not find torch.load pattern to patch.")
        sys.exit(1)

    with open(io_path, "w") as f:
        f.write(patched)

    print(f"Patched {count} torch.load() calls in {io_path}")


if __name__ == "__main__":
    patch()
