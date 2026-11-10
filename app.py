"""
Resume Match Analyzer - Streamlit UI.

All the logic lives in src/. This file only handles input, display, and
the edge cases a real user will hit.

Run locally:   streamlit run app.py
"""

import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import WEIGHTS
from src.parser import parse_resume
from src.scorer import COMPONENT_LABELS, load_model, score_resume

SAMPLE_RESUME = Path(__file__).parent / "docs" / "sample_resume.pdf"
SAMPLE_POSTING = Path(__file__).parent / "docs" / "sample_posting.txt"

st.set_page_config(page_title="Resume Match Analyzer", page_icon="📄", layout="centered")


# ---------------------------------------------------------------------------
# Model loading - cached
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the language model (first run only)...")
def get_model():
    # WHY cache_resource: Streamlit re-runs this whole script on EVERY
    # interaction (each click, each keystroke in a widget). Without caching,
    # the ~440 MB model would reload every time, taking seconds per click.
    # cache_resource loads it once per server process and shares it.
    return load_model()


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def chips(skills, kind):
    """Render skills as coloured pills. kind = 'match' (green) or 'missing' (amber)."""
    if not skills:
        return "<span style='opacity:0.6'>None</span>"
    color = ("rgba(34,139,34,0.18)", "#1e7b34") if kind == "match" else ("rgba(230,145,0,0.20)", "#b26a00")
    style = (f"display:inline-block;margin:2px 4px 2px 0;padding:2px 10px;border-radius:12px;"
             f"background:{color[0]};border:1px solid {color[1]};font-size:0.85rem")
    # Skill names come from our own fixed vocabulary, never from user
    # input, so inserting them into HTML is safe.
    return "".join(f"<span style='{style}'>{s}</span>" for s in skills)


def overlap_label(score):
    # Describes the TEXT overlap with ONE posting, never the person.
    # Deliberately no words like "qualified", "poor" or "reject".
    if score >= 60:
        return "Strong overlap with this posting"
    if score >= 35:
        return "Partial overlap with this posting"
    return "Limited overlap with this posting"


# ---------------------------------------------------------------------------
# Header and expectations
# ---------------------------------------------------------------------------
st.title("📄 Resume Match Analyzer")
st.write("For **job seekers**: see how your resume's wording lines up with a job "
         "posting you're interested in, and get specific suggestions to present your "
         "real experience more clearly.")

with st.expander("How this works and what it can't do", expanded=False):
    st.markdown(f"""
**What it does.** Compares your resume with one job posting in four ways:

| Part | Weight | How |
|---|---|---|
| Skills match | {WEIGHTS['skills']:.0%} | Looks for the posting's tech skills (from a ~240-item list) in your resume |
| Experience match | {WEIGHTS['experience']:.0%} | Language-model similarity: your Experience/Projects section vs the posting's responsibilities |
| Education match | {WEIGHTS['education']:.0%} | Similarity between your field of study and the posting's education requirements |
| Overall fit | {WEIGHTS['overall']:.0%} | Similarity of the whole resume vs the whole posting |

**What it can't do**
- **The weights are judgement calls**, not learned from hiring data. Nobody has validated
  that 40/30/15/15 is "right".
- It measures **topic overlap, not ability**. It can't tell whether you used a tool once or
  for years, and it doesn't check seniority or years of experience.
- The skill list is **tech-focused**. Postings in other fields mostly can't be skill-matched.
- PDF reading can fail on **multi-column or heavily designed** resumes, and **scanned**
  (image) PDFs can't be read at all without OCR.
- A real applicant tracking system (ATS) works differently and varies by employer.
  This is an estimate against one posting, nothing more.
""")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# A demo nobody can try without uploading a personal document is a demo
# most people close. The sample resume is for an invented person. One click
# runs the whole analysis on it, regardless of anything already uploaded.
sample_clicked = st.button("Try the sample (synthetic resume + posting)")
if sample_clicked:
    # Must happen BEFORE the text area below is created: Streamlit doesn't
    # allow changing a widget's value after it has been drawn in this run.
    st.session_state["posting_text"] = SAMPLE_POSTING.read_text(encoding="utf-8")

left, right = st.columns(2)
with left:
    uploaded = st.file_uploader("Your resume (PDF)", type=["pdf"])
with right:
    posting_text = st.text_area("Job posting (paste the text)", height=230,
                                key="posting_text",
                                placeholder="Paste the full job description here...")

analyse = st.button("Analyse", type="primary", width="stretch")

st.caption("🔒 **Privacy:** your PDF is read in memory only. It is never written to disk, "
           "logged, or sent to any other service. The results are held in memory for your "
           "browser session and discarded when the session ends.")


# ---------------------------------------------------------------------------
# Run the analysis
# ---------------------------------------------------------------------------
def run_analysis(use_sample: bool):
    # --- Edge case: missing inputs -> friendly message, no crash
    if uploaded is None and not use_sample:
        st.warning("Please upload your resume as a PDF (or try the sample).")
        return None
    if not posting_text or len(posting_text.strip()) < 50:
        st.warning("Please paste the job posting text (at least a few lines).")
        return None

    # --- Read the PDF entirely in memory.
    # PRIVACY: getvalue() gives the bytes Streamlit already holds in RAM, and
    # BytesIO wraps them as a file-like object. pdfplumber reads from that
    # directly, so the upload never touches the disk, not even as a temp file.
    if use_sample:
        pdf_bytes = io.BytesIO(SAMPLE_RESUME.read_bytes())
    else:
        pdf_bytes = io.BytesIO(uploaded.getvalue())

    try:
        resume = parse_resume(pdf_bytes)
    except Exception:
        # Deliberately not showing or logging the exception details: they
        # can include fragments of the file's contents.
        st.error("Sorry, that PDF couldn't be read. It may be damaged, password-protected, "
                 "or not a real PDF. Try re-exporting it from your word processor.")
        return None

    # --- Edge case: scanned / image-only PDF
    if resume.looks_scanned:
        st.error("This PDF contains almost no selectable text. It looks like a scanned "
                 "document or an image. This tool can't read images; it would need OCR "
                 "(optical character recognition). Try exporting your resume from Word or "
                 "Google Docs as a PDF instead of scanning or 'printing to image'.")
        return None

    with st.spinner("Comparing your resume with the posting..."):
        return resume, score_resume(get_model(), resume, posting_text)


if analyse or sample_clicked:
    outcome = run_analysis(use_sample=sample_clicked)
    st.session_state["outcome"] = outcome
    st.session_state["animate"] = outcome is not None
    st.session_state["is_sample"] = sample_clicked


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
outcome = st.session_state.get("outcome")
if outcome:
    resume, result = outcome
    st.divider()
    if st.session_state.get("is_sample"):
        st.info("Showing results for the **synthetic sample**: Alex Rivera is an invented "
                "person, and the posting is for a fictional company.")

    if result.low_confidence:
        st.warning("**Low confidence.** No standard resume sections (Experience, Education, "
                   "Skills) were detected. This may not be a resume, or its layout couldn't "
                   "be read. Treat the numbers below with caution.")

    # --- Overall score, animated
    st.subheader("Match estimate")
    score_slot = st.empty()
    bar = st.progress(0)
    if st.session_state.get("animate"):
        # Short count-up animation. Only on a fresh analysis, not on every rerun.
        steps = 25
        for i in range(1, steps + 1):
            value = result.score * i / steps
            score_slot.markdown(f"<h1 style='margin:0'>{value:.0f}<span style='font-size:1.2rem;"
                                f"opacity:0.6'> / 100</span></h1>", unsafe_allow_html=True)
            bar.progress(int(value))
            time.sleep(0.02)
        st.session_state["animate"] = False
    score_slot.markdown(f"<h1 style='margin:0'>{result.score:.0f}<span style='font-size:1.2rem;"
                        f"opacity:0.6'> / 100</span></h1>", unsafe_allow_html=True)
    bar.progress(int(result.score))
    st.write(f"**{overlap_label(result.score)}**")

    st.info("ℹ️ **This is an estimate**, not a verdict. The weights behind it are judgement "
            "calls, it measures wording overlap rather than ability, and a real ATS works "
            "differently. Use it to spot gaps in how your resume *reads*, not to decide "
            "whether to apply.")

    # --- Where the score comes from
    st.subheader("Where the score comes from")
    for key, c in result.components.items():
        label = f"{COMPONENT_LABELS[key]} - weight {c.weight:.0%}"
        if c.available:
            if abs(c.effective_weight - c.weight) > 0.005:
                label += f" (counted as {c.effective_weight:.0%}; see notes)"
            st.progress(int(round(c.score)), text=f"{label}: **{c.score:.0f}**")
        else:
            st.progress(0, text=f"{label}: *not measured* (weight shared among the others)")

    # --- Skills
    st.subheader("Skills the posting mentions")
    st.markdown("**Found in your resume**", help="Exact or alias matches, e.g. sklearn = scikit-learn")
    st.markdown(chips(result.matched_skills, "match"), unsafe_allow_html=True)
    st.markdown("**Not found in your resume**", help="Most-mentioned in the posting first")
    st.markdown(chips([s for s, _ in result.missing_skills], "missing"), unsafe_allow_html=True)
    if not result.components["skills"].available:
        st.caption("The posting doesn't use any skills from this tool's tech-focused list, "
                   "so skills couldn't be compared.")

    # --- Recommendations
    st.subheader("Suggestions")
    order = {"high": 0, "medium": 1, "low": 2}
    icons = {"high": "🔶", "medium": "🔸", "low": "▫️"}
    for i, rec in enumerate(sorted(result.recommendations, key=lambda r: order[r["priority"]]), 1):
        st.markdown(f"{i}. {icons[rec['priority']]} {rec['text']}")

    if result.notes:
        st.subheader("Notes")
        for note in result.notes:
            st.caption("• " + note)

    # --- Raw numbers
    with st.expander("Raw numbers"):
        rows = []
        for key, c in result.components.items():
            rows.append({
                "component": COMPONENT_LABELS[key],
                "nominal weight": f"{c.weight:.0%}",
                "weight used": f"{c.effective_weight:.0%}",
                "raw cosine": None if c.raw is None else round(c.raw, 3),
                "calibrated score": round(c.score, 1) if c.available else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("Sections detected in your resume:",
                 ", ".join(resume.found) or "none",
                 "| not detected:", ", ".join(resume.missing) or "none")
        st.write(f"Text extracted: {len(resume.text):,} characters")
        if result.missing_keywords:
            st.write("Posting words absent from resume (count):",
                     ", ".join(f"{w} ({n})" for w, n in result.missing_keywords))
        st.caption("Raw cosine similarity is rescaled so that text from an unrelated field "
                   "maps to about 0 and a strongly tailored resume to about 100. "
                   "See docs/scoring_design.md for the measurements.")
