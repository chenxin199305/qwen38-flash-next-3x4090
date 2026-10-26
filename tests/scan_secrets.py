#!/usr/bin/env python3
"""Fail if obvious private deployment data is committed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", "LICENSE"}
PATTERNS = [
    (re.compile(r"192\.168\.\d+\.\d+"), "private LAN IP"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "Hugging Face token"),
    (re.compile(r"ssh-rsa\s+AAAA"), "SSH private key material"),
    (re.compile(r"(?i)password\s*=\s*\S+"), "plaintext password"),
    (re.compile(r"VLLM_API_KEY\s*=\s*[^\s\"']+"), "inline API key"),
]


def main() -> int:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == "scan_secrets.py" or any(part in SKIP for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern, label in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    if hits:
        print("secret scan failed:")
        print("\n".join(hits))
        return 1
    print("secret scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())