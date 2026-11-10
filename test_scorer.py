"""
Phase 2 check: score a real resume against real postings and print the full
breakdown plus recommendations.

Test design:
  - Two ML/data postings a fresh ML grad would apply to (expect moderate-high)
  - One senior role (expect lower)
  - One adjacent-field posting (data engineering - expect in between)
  - One UNRELATED posting held out from calibration (expect near zero).
    Held out matters: the floor was set from the nursing and accounting
    postings, so those score ~0 by construction. Testing on them would
    prove nothing.

Usage:  python test_scorer.py ["data\\my resume.pdf"]
"""

import glob
import sys
import textwrap

from src.parser import parse_resume
from src.postings import load_postings
from src.scorer import COMPONENT_LABELS, load_model, score_resume


def print_result(label, title, result):
    print("=" * 78)
    print(f"{label}: {title}")
    print("=" * 78)
    print(f"MATCH SCORE: {result.score:.1f} / 100"
          + ("   [LOW CONFIDENCE]" if result.low_confidence else ""))
    print(f"\n  {'Component':24} {'weight':>7} {'used':>6} {'score':>7} {'raw cos':>8}")
    for key, c in result.components.items():
        raw = f"{c.raw:.3f}" if c.raw is not None else "-"
        score = f"{c.score:.1f}" if c.available else "n/a"
        print(f"  {COMPONENT_LABELS[key]:24} {c.weight:7.0%} {c.effective_weight:6.0%} "
              f"{score:>7} {raw:>8}")
    print(f"\n  Matched skills ({len(result.matched_skills)}): "
          f"{', '.join(result.matched_skills) or '-'}")
    print(f"  Missing skills ({len(result.missing_skills)}): "
          f"{', '.join(f'{s} x{n}' if n > 1 else s for s, n in result.missing_skills) or '-'}")
    for note in result.notes:
        print(textwrap.fill(f"NOTE: {note}", 76, initial_indent="  ", subsequent_indent="        "))
    print("\n  Recommendations:")
    for r in result.recommendations:
        print(textwrap.fill(f"[{r['priority']}] {r['text']}", 76,
                            initial_indent="  - ", subsequent_indent="    "))
    print()


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else sorted(p for p in glob.glob("data/*.pdf") if "sample_" not in p)[0]
    resume = parse_resume(pdf)
    postings = load_postings("data/ml_and_data_job_postings.txt")
    heldout = load_postings("data/heldout_unrelated_posting.txt")[0]

    cases = [
        ("APPLY #1", postings[0]),
        ("APPLY #2", postings[1]),
        ("SENIOR", postings[2]),
        ("ADJACENT", postings[3]),
        ("UNRELATED (held out)", heldout),
    ]

    print("Loading model...")
    model = load_model()
    results = []
    for label, posting in cases:
        result = score_resume(model, resume, posting["text"])
        print_result(label, posting["title"], result)
        results.append((label, posting["title"], result.score))

    print("=" * 78)
    print("RANKING")
    print("=" * 78)
    for label, title, score in sorted(results, key=lambda r: -r[2]):
        print(f"  {score:5.1f}  {label:22} {title}")

    # The non-negotiable sanity check from the design: the unrelated posting
    # must score clearly lowest. If not, the scoring is broken.
    unrelated = results[-1][2]
    others = [s for _, _, s in results[:-1]]
    margin = min(others) - unrelated
    ok = margin >= 10
    print(f"\n  Unrelated posting: {unrelated:.1f}. Lowest related posting: {min(others):.1f}. "
          f"Margin: {margin:.1f} points -> {'OK' if ok else 'BROKEN: fix before moving on'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
