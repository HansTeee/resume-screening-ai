"""
Phase 2 measurement: compare two embedding models on real data.

Reports, for each model:
  - load time and size on disk
  - time to embed everything the scorer needs for one resume vs one posting
  - RAW cosine similarity (before calibration) for every posting, per
    component, so we can see (a) whether the ranking is sensible and
    (b) what an unrelated-field posting scores: the "floor".

Usage:  python compare_models.py
"""

import glob
import os
import time

from src.parser import parse_resume
from src.postings import load_postings
import torch

from src.scorer import cosine, degree_field, load_model, split_posting

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


def model_size_mb(name: str) -> float:
    """Size of the model's files in the Hugging Face cache."""
    # Without symlinks (the Windows default), files live in snapshots/
    # rather than blobs/, so walk the whole model folder.
    folder = "models--" + name.replace("/", "--")
    root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", folder)
    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, files in os.walk(root) for f in files)
    return total / 1e6


def main():
    # A free Hugging Face CPU Space has 2 vCPUs. Limiting torch to 2 threads
    # makes the timings closer to what the deployed app will see.
    torch.set_num_threads(2)
    resume = parse_resume(sorted(p for p in glob.glob("data/*.pdf") if "sample_" not in p)[0])
    postings = load_postings("data/ml_and_data_job_postings.txt")
    unrelated = load_postings("data/unrelated_postings.txt")
    for p in unrelated:
        p["title"] = "[UNRELATED] " + p["title"]
    everything = postings + unrelated

    for name in MODELS:
        print("=" * 78)
        print(name)
        print("=" * 78)
        t0 = time.perf_counter()
        model = load_model(name)
        load_s = time.perf_counter() - t0
        model.encode(["warm-up"])  # first call has one-off setup cost
        print(f"Load time: {load_s:.1f}s   Size on disk: {model_size_mb(name):.0f} MB   "
              f"Dim: {len(model.encode('x'))}")

        print(f"\n{'Posting':52} {'exp':>6} {'edu':>6} {'whole':>6}  {'time':>6}")
        for p in everything:
            parts = split_posting(p["text"])
            t0 = time.perf_counter()
            exp = cosine(model, resume.sections["EXPERIENCE"], parts["responsibilities"])
            edu = (cosine(model, degree_field(resume.sections["EDUCATION"]),
                          degree_field(parts["education"]))
                   if parts["education"] else float("nan"))
            whole = cosine(model, resume.text, p["text"])
            elapsed = time.perf_counter() - t0
            print(f"{p['title'][:52]:52} {exp:6.3f} {edu:6.3f} {whole:6.3f}  {elapsed:5.2f}s")
        print()


if __name__ == "__main__":
    main()
