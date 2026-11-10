"""
Central configuration for the resume match analyzer.

WHY a config file: every "magic number" and keyword list lives here, so when
you want to tweak a weight or add a heading variation you change ONE place,
and the rest of the code doesn't need to know the details.
"""

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Chosen in Phase 2 after measuring both candidates on real data.
# See docs/scoring_design.md ("Model choice") for the numbers.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Kept so the comparison script can re-run the benchmark.
ALTERNATIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Scoring weights (must add up to 1.0)
# ---------------------------------------------------------------------------
# These are JUDGEMENT CALLS, not learned from data. docs/scoring_design.md
# explains the reasoning behind each one.
WEIGHTS = {
    "skills": 0.40,
    "experience": 0.30,
    "education": 0.15,
    "overall": 0.15,
}

# ---------------------------------------------------------------------------
# Similarity calibration (measured in Phase 2, all-mpnet-base-v2)
# ---------------------------------------------------------------------------
# Raw cosine similarity between two unrelated English texts is NOT ~0,
# because they share ordinary words and grammar. So each semantic component
# is rescaled:
#     displayed = (raw - floor) / (ceiling - floor), clipped to 0..1
#
# floor   = the HIGHER raw score of two unrelated-field postings (nursing,
#           accounting) against a tech resume. Unrelated text maps to ~0.
# ceiling = what a synthetic resume deliberately tailored to a posting
#           scored (experience 0.785 / 0.708, whole 0.737 / 0.706), or for
#           education, a degree field the posting explicitly lists
#           (0.41-0.445). Real resumes rarely exceed these.
#
# Floors differ a lot by component, which is why there isn't one number:
#   experience: nursing 0.043, accounting 0.085
#   education:  nursing 0.228, accounting 0.275  (short texts -> high floor)
#   overall:    nursing 0.123, accounting 0.203
# If you change EMBEDDING_MODEL, these MUST be re-measured
# (python compare_models.py) - each model has its own similarity scale.
CALIBRATION = {
    "experience": {"floor": 0.085, "ceiling": 0.75},
    "education":  {"floor": 0.275, "ceiling": 0.45},
    "overall":    {"floor": 0.203, "ceiling": 0.72},
}

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------
# A line is treated as a section heading if, after removing trailing dates
# and a trailing colon, the WHOLE line equals one of these phrases.
# Matching is case-insensitive.
#
# Judgement call: PROJECTS is mapped to EXPERIENCE. For students and fresh
# grads, projects ARE the relevant experience; leaving them in OTHER would
# make the experience score unfairly low for exactly the people this tool
# is meant to help.
SECTION_HEADINGS = {
    "EXPERIENCE": [
        "experience", "experiences", "work experience", "professional experience",
        "employment history", "employment", "work history", "relevant experience",
        "internships", "internship", "projects", "academic projects",
        "personal projects", "selected projects", "research experience",
        "leadership experience",
    ],
    "EDUCATION": [
        "education", "educational background", "academic background",
        "academic qualifications", "qualifications", "academics",
    ],
    "SKILLS": [
        "skills", "technical skills", "core skills", "key skills",
        "skills and tools", "skills & tools", "technologies", "tech stack",
        "core competencies", "competencies", "tools", "technical proficiencies",
    ],
    "OTHER": [
        "summary", "professional summary", "profile", "objective",
        "career objective", "about me", "certifications", "certificates",
        "awards", "honors", "honours", "achievements", "publications",
        "languages", "interests", "hobbies", "references", "volunteer",
        "volunteering", "activities", "extracurricular activities",
        "organizations", "affiliations", "seminars", "trainings",
    ],
}

# Headings longer than this many words are probably sentences, not headings.
MAX_HEADING_WORDS = 5

# If extracted text is shorter than this, the PDF is probably a scanned image.
MIN_TEXT_CHARS = 100
