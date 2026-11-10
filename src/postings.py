"""
Load job postings from a plain-text file.

The file format is a series of blocks separated by a "JOB POSTING <n>"
banner (see data/ml_and_data_job_postings.txt). This only exists for the
test scripts; the app takes a pasted posting directly.
"""

import re

_BANNER = re.compile(r"^=+\s*\nJOB POSTING \d+\s*\n=+\s*$", re.MULTILINE)


def load_postings(path: str) -> list:
    """Return a list of {"title": ..., "text": ...} dicts."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    postings = []
    for block in _BANNER.split(raw):
        block = block.strip()
        if not block:
            continue
        match = re.search(r"^Job Title:\s*(.+)$", block, re.MULTILINE)
        title = match.group(1).strip() if match else block.splitlines()[0]
        postings.append({"title": title, "text": block})
    return postings
