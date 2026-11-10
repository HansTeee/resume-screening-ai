"""
Phase 1 check: run the parser and skill extractor on real files and print
what came out, so a human can judge whether the extraction is trustworthy.

Usage (from the project folder):
    python test_parser.py                          (uses the first PDF in data/)
    python test_parser.py "data\\my resume.pdf"
"""

import glob
import sys

from src.parser import parse_resume
from src.postings import load_postings
from src.skills import extract_skills

POSTINGS_FILE = "data/ml_and_data_job_postings.txt"


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdfs = sorted(p for p in glob.glob("data/*.pdf") if "sample_" not in p)
        if not pdfs:
            print("No PDF found in data/. Put your resume there or pass a path.")
            return
        pdf_path = pdfs[0]

    resume = parse_resume(pdf_path)

    print("=" * 70)
    print("EXTRACTED TEXT")
    print("=" * 70)
    print(f"Length: {len(resume.text)} characters")
    if resume.looks_scanned:
        print("WARNING: almost no text - this looks like a scanned/image PDF "
              "and needs OCR. Results below are not meaningful.")
    print("First 500 characters:")
    print("-" * 70)
    print(resume.text[:500])
    print("-" * 70)

    print("\n" + "=" * 70)
    print("SECTIONS")
    print("=" * 70)
    for name, content in resume.sections.items():
        status = "found" if name in resume.found or name == "OTHER" else "NOT FOUND"
        print(f"  {name:11} {status:10} {len(content):5} chars")
    if resume.missing:
        print(f"  Missing: {', '.join(resume.missing)}. The scorer will "
              "redistribute their weight rather than score them as zero.")

    print("\n" + "=" * 70)
    print("SKILLS IN RESUME")
    print("=" * 70)
    resume_skills = extract_skills(resume.text)
    print(f"  {len(resume_skills)} found: {', '.join(resume_skills)}")

    print("\n" + "=" * 70)
    print("SKILLS IN EACH JOB POSTING")
    print("=" * 70)
    for i, posting in enumerate(load_postings(POSTINGS_FILE), start=1):
        skills = extract_skills(posting["text"])
        print(f"\n  [{i}] {posting['title']}")
        print(f"      {len(skills)} found: {', '.join(skills)}")

    print("\n" + "=" * 70)
    print("POSSIBLY MANGLED TEXT")
    print("=" * 70)
    if resume.warnings:
        for w in resume.warnings:
            print("  -", w)
    else:
        print("  None detected by the heuristics (repeated chars, missing "
              "spaces, letter-spacing, unmapped glyphs).")
        print("  Note: scrambled READING ORDER can't be detected automatically;")
        print("  compare the section lengths above against the real resume.")


if __name__ == "__main__":
    main()
