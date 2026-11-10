"""
Does following the tool's advice actually raise the score?

Scores the SYNTHETIC sample resume against the sample posting, prints the
advice, then scores an "after" version in which the same (invented) person
NAMES things their resume already DESCRIBED (container -> Docker, cloud
VM -> AWS EC2, Python web API -> FastAPI). No new experience is added. See
scripts/make_sample_resume.py.

Usage:  python scripts/make_sample_resume.py   (once, to build the PDFs)
        python test_advice_loop.py
"""

from src.parser import parse_resume
from src.scorer import COMPONENT_LABELS, load_model, score_resume

posting = open("docs/sample_posting.txt", encoding="utf-8").read()
model = load_model()

before = score_resume(model, parse_resume("docs/sample_resume.pdf"), posting)
after = score_resume(model, parse_resume("data/sample_resume_after.pdf"), posting)

print("BEFORE advice:", before.score)
print("  missing skills:", ", ".join(f"{s} x{n}" if n > 1 else s for s, n in before.missing_skills))
print("  top recommendation:", before.recommendations[0]["text"][:160], "...")
print("AFTER  advice:", after.score)
print("  missing skills:", ", ".join(f"{s} x{n}" if n > 1 else s for s, n in after.missing_skills))
print()
print(f"{'component':24} {'before':>7} {'after':>7}")
for key in before.components:
    b, a = before.components[key], after.components[key]
    print(f"{COMPONENT_LABELS[key]:24} {b.score:7.1f} {a.score:7.1f}")
print(f"{'TOTAL':24} {before.score:7.1f} {after.score:7.1f}   ({after.score - before.score:+.1f})")
