"""
Hybrid scoring engine: keyword skills match + embedding similarity.

See docs/scoring_design.md for WHY each component exists and why the
weights are what they are (short version: they're judgement calls).
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CALIBRATION, EMBEDDING_MODEL, WEIGHTS
from src.parser import ParsedResume
from src.skills import count_skills, extract_skills

COMPONENT_LABELS = {
    "skills": "Skills match",
    "experience": "Experience match",
    "education": "Education match",
    "overall": "Overall semantic fit",
}


# ---------------------------------------------------------------------------
# Model loading and embedding
# ---------------------------------------------------------------------------
def load_model(name: str = EMBEDDING_MODEL):
    """
    Load a sentence-transformers model on CPU.

    Imported inside the function so that modules which only need keyword
    matching (and the tests for them) don't pay the multi-second cost of
    importing torch.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def _chunk(text: str, max_words: int = 120) -> list:
    """
    Split text into chunks of whole lines, each at most ~max_words words.

    WHY: these models silently TRUNCATE input (MiniLM at 256 tokens). On a
    real resume the experience section was 406 tokens, so without chunking
    the model never saw the last ~37% of it. 120 words is roughly 160-200
    tokens, safely under the limit.
    """
    chunks, current, count = [], [], 0
    for line in text.splitlines():
        words = len(line.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(line)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed(model, text: str) -> np.ndarray:
    """
    One vector for a piece of text of any length.

    Each chunk is embedded separately, then the chunk vectors are averaged
    and re-normalised to unit length. Unit length means a plain dot product
    between two vectors IS their cosine similarity.
    """
    chunks = _chunk(text) or [""]
    vectors = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    mean = vectors.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-12)


def cosine(model, a: str, b: str) -> float:
    return float(np.dot(embed(model, a), embed(model, b)))


def calibrate(raw: float, component: str) -> float:
    """
    Map raw cosine onto 0-100.

    WHY: unrelated English texts still have positive cosine similarity
    (shared grammar and common words), so raw * 100 squeezes every result
    into a narrow, misleadingly high band. We measured what an unrelated
    posting scores (the floor) and map that to 0. See scoring_design.md.
    """
    floor = CALIBRATION[component]["floor"]
    ceiling = CALIBRATION[component]["ceiling"]
    scaled = (raw - floor) / (ceiling - floor)
    return float(np.clip(scaled, 0.0, 1.0) * 100)


# ---------------------------------------------------------------------------
# Splitting a job posting into its parts
# ---------------------------------------------------------------------------
_RESP_WORDS = re.compile(r"responsibilit|duties|what you.?ll do|job description|the role", re.I)
_REQ_WORDS = re.compile(r"qualification|requirement|what we.?re looking for|who you are|must have", re.I)
# Lines that talk about education. Used to pick out the part of the posting
# the EDUCATION section should be compared against.
_EDU_WORDS = re.compile(
    r"\b(degree|bachelor'?s?|master'?s?|ph\.?d|diploma|graduate|university|college|"
    r"b\.?s\.?|m\.?s\.?|course|major|field of study|education)\b", re.I)


# Generic degree vocabulary. Removed before comparing education text, so
# that only the FIELD of study is compared.
# WHY (measured): with the full lines, "Bachelor of Science in Nursing"
# scored MORE similar to "BS Computer Engineering, University of ..." than
# "Bachelor's degree in Math, Statistics, ... Engineering" did, under both
# models. Short degree lines all share the same template, and the template
# swamped the field of study. Stripping it fixed the ranking.
_DEGREE_BOILERPLATE = re.compile(
    # Longer phrases first, so "Bachelor of Science" is removed as a unit but
    # "Science" in "Computer Science" survives.
    r"\b((bachelor|master)'?s?\s+of\s+(science|arts)|"
    r"bachelor'?s?|master'?s?|doctorate|degree|diploma|of|in|a|an|"
    r"university|college|school|institute|education|the|bs|ba|ms|ma|minimum|"
    r"at least|or|and|related|field|fields|preferred|required|graduate|"
    r"(19|20)\d{2}|present|expected|etc)\b",
    re.I)


def degree_field(text: str) -> str:
    """'BS Computer Engineering\\nUniversity of X\\n2022-2026' -> 'Computer Engineering'"""
    # Drop the parts naming an institution: WHERE someone studied says
    # nothing about WHAT they studied, and university names add noise.
    # Split on commas, pipes and " - " first, because resumes often put the
    # degree and the school on ONE line ("BS Computer Science, X University,
    # 2021-2025"); dropping the whole line would lose the degree too.
    segments = re.split(r",|\||\s-\s|\n", text)
    kept = [seg for seg in segments
            if not re.search(r"\b(university|college|institute|school)\b", seg, re.I)]
    text = _DEGREE_BOILERPLATE.sub(" ", " ".join(kept))
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def resume_degree_lines(education_section: str) -> str:
    """
    Keep only the lines of a resume's education section that name a degree.

    WHY (measured): on the sample resume, a thesis title in the education
    section ("Classifying customer support tickets...") diluted the field of
    study so much that "Computer Science" scored 17/100 against a posting
    asking for Computer Science. The education component checks the DEGREE
    requirement, so the thesis/coursework lines are left to the other
    components. Falls back to the whole section if no degree line is found.
    """
    lines = [l for l in education_section.splitlines() if _EDU_WORDS.search(l)]
    return "\n".join(lines) if lines else education_section


def split_posting(text: str) -> dict:
    """
    Best-effort split into responsibilities / requirements / education lines.

    Postings have no standard format, so every part falls back to the whole
    posting if we can't find it. A failed split degrades the comparison
    rather than breaking it.
    """
    resp, req, current = [], [], None
    for line in text.splitlines():
        if _RESP_WORDS.search(line) and len(line.split()) <= 6:
            current = resp
            continue
        if _REQ_WORDS.search(line) and len(line.split()) <= 6:
            current = req
            continue
        if current is not None:
            current.append(line)
    education = [l for l in text.splitlines() if _EDU_WORDS.search(l)]
    return {
        "responsibilities": "\n".join(resp).strip() or text,
        "requirements": "\n".join(req).strip() or text,
        "education": "\n".join(education).strip(),  # "" = posting doesn't mention it
        "split_ok": bool(resp and req),
    }


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class Component:
    name: str
    available: bool
    score: float = 0.0          # 0-100, calibrated
    raw: float | None = None    # raw cosine, for the "raw numbers" view
    weight: float = 0.0         # nominal weight from config
    effective_weight: float = 0.0  # after redistributing missing components
    note: str = ""


@dataclass
class MatchResult:
    score: float
    components: dict
    matched_skills: list
    missing_skills: list         # [(skill, mentions_in_posting)], most-mentioned first
    extra_skills: list           # in resume, not asked for by posting
    missing_keywords: list       # [(word, mentions)] non-skill words absent from resume
    recommendations: list
    low_confidence: bool
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# The four components
# ---------------------------------------------------------------------------
def skills_component(resume_text: str, posting_text: str):
    """
    Keyword-based. Uses the WHOLE resume, not just the SKILLS section, so a
    missed "Skills" heading doesn't hide skills mentioned under projects.
    """
    required = count_skills(posting_text)
    have = set(extract_skills(resume_text))
    matched = sorted((s for s in required if s in have), key=lambda s: (-required[s], s))
    missing = sorted(((s, n) for s, n in required.items() if s not in have),
                     key=lambda x: (-x[1], x[0]))
    extra = sorted(have - set(required), key=str.lower)

    if not required:
        comp = Component("skills", available=False, weight=WEIGHTS["skills"],
                         note="The posting lists no skills from this tool's (tech-focused) "
                              "vocabulary, so this part can't be measured.")
    else:
        comp = Component("skills", available=True, weight=WEIGHTS["skills"],
                         score=100 * len(matched) / len(required),
                         note=f"{len(matched)} of {len(required)} skills the posting mentions "
                              "appear in your resume.")
    return comp, matched, missing, extra


def semantic_component(model, name: str, resume_part: str, posting_part: str,
                       missing_note: str) -> Component:
    if not resume_part.strip() or not posting_part.strip():
        return Component(name, available=False, weight=WEIGHTS[name], note=missing_note)
    raw = cosine(model, resume_part, posting_part)
    return Component(name, available=True, weight=WEIGHTS[name],
                     raw=raw, score=calibrate(raw, name))


# ---------------------------------------------------------------------------
# Keywords that aren't in the skill vocabulary
# ---------------------------------------------------------------------------
# Words common in postings that carry no information about the job itself.
_POSTING_BOILERPLATE = {
    "experience", "ability", "able", "strong", "skills", "skill", "knowledge",
    "work", "working", "using", "including", "etc", "years", "year", "minimum",
    "qualifications", "responsibilities", "description", "job", "title",
    "company", "location", "source", "preferred", "related", "field", "plus",
    "understanding", "background", "proficiency", "excellent", "good", "solid",
    "areas", "like", "particularly", "applicable", "set", "new", "team",
    "philippines", "inc", "corporation", "hybrid", "remote", "https", "www",
    "com", "jobs", "kalibrr", "least", "hands", "large", "variety", "highly",
    "ability", "use", "usage", "provide", "ensure", "based", "via",
}


def missing_keywords(resume_text: str, posting_text: str, skill_words: set,
                     top: int = 10, min_count: int = 2):
    """
    Content words that appear in the posting but nowhere in the resume.

    Skips words already covered by the skills check, stop words, and
    posting boilerplate. Only words the posting uses at least `min_count`
    times are kept. WHY: on real postings, words used once were mostly
    filler ("utilizing", "combination"), while repeated words were the
    posting's actual themes. Returns [(word, count)] most-frequent first.
    """
    # Fold simple plurals so "models" in a posting counts as present when the
    # resume says "model". Crude (no real stemming), but it removed the most
    # obvious false "missing word" on real data.
    def fold(w):
        return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w

    resume_words = {fold(w) for w in re.findall(r"[a-z][a-z\-]+", resume_text.lower())}
    # Only the descriptive part of a posting, not the header lines with
    # company name, location and URL.
    body = "\n".join(l for l in posting_text.splitlines()
                     if not re.match(r"^(job title|company|location|source)\s*:", l, re.I))
    counts = Counter(
        fold(w) for w in re.findall(r"[a-z][a-z\-]+", body.lower())
        if len(w) > 3 and w not in ENGLISH_STOP_WORDS and w not in _POSTING_BOILERPLATE
        and w not in skill_words
    )
    return [(w, n) for w, n in counts.most_common()
            if w not in resume_words and n >= min_count][:top]


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def build_recommendations(components, missing, extra, keywords, resume: ParsedResume) -> list:
    """
    Turn the numbers into suggestions.

    Rules for wording (deliberate):
      - Suggestions to the candidate, never judgements about them.
      - Every "add X" is conditional on the candidate actually having X.
        The tool helps present real experience; it must never encourage
        claiming skills someone doesn't have.
    """
    recs = []

    # 1. Missing skills, most-mentioned first. Prominence is the best signal
    #    we have of what the employer cares about.
    if missing:
        top = missing[:6]
        listed = ", ".join(f"{s} (mentioned {n}x)" if n > 1 else s for s, n in top)
        recs.append({
            "priority": "high",
            "text": f"The posting mentions skills your resume doesn't: {listed}. "
                    "If you have genuine experience with any of these, consider naming "
                    "them explicitly. Keyword filters often look for the exact term. "
                    "If you don't, leave them out; this list is a gap map, not a to-do list.",
        })

    # 2. Components scoring well below the others.
    # "Notably below" = 25+ points under the AVERAGE of the other available
    # components. (Comparing to the single best one over-triggers: one
    # near-100 component would make everything else look "low".)
    avail = {k: c for k, c in components.items() if c.available}
    if len(avail) >= 3:
        for key, c in avail.items():
            others = [o.score for k, o in avail.items() if k != key]
            if sum(others) / len(others) - c.score >= 25:
                recs.append({"priority": "medium", "text": _section_advice(key, c)})

    # 3. Sections we couldn't find.
    for section in resume.missing:
        recs.append({
            "priority": "medium",
            "text": f"No {section.title()} heading was detected. If your resume has one "
                    f"under a different name, a standard heading like \"{section.title()}\" "
                    "helps automated parsers (including real ATS) find it.",
        })

    # 4. Posting vocabulary the resume never uses.
    if keywords:
        words = ", ".join(w for w, _ in keywords[:8])
        recs.append({
            "priority": "low",
            "text": f"Words from the posting that don't appear in your resume: {words}. "
                    "Where these genuinely describe work you've done, using the posting's "
                    "own wording can make the match clearer to both people and software.",
        })

    # 5. Resume skills the posting doesn't ask about. Not a problem in
    #    itself; only worth mentioning when there are many.
    # Skipped when the posting asks for no recognisable skills at all: then
    # "move the ones the posting asks for" would be meaningless.
    if len(extra) >= 8 and components["skills"].available:
        recs.append({
            "priority": "low",
            "text": f"Your resume lists {len(extra)} skills this posting doesn't mention "
                    f"(e.g. {', '.join(extra[:5])}). They're worth keeping, but for this "
                    "application you might move the ones the posting asks for nearer the top.",
        })

    if not recs:
        recs.append({"priority": "low",
                     "text": "No major gaps found against this posting's wording."})
    return recs


def _section_advice(key: str, c: Component) -> str:
    if key == "skills":
        return ("Skills match is lower than the other components. The posting's "
                "specific tools and terms are the quickest thing to check.")
    if key == "experience":
        return ("Your experience section reads as less related to this posting's "
                "responsibilities than the rest of your resume. Consider whether your "
                "bullet points describe the kind of work the posting lists, in similar terms.")
    if key == "education":
        return ("Education matches this posting less closely than other parts. If you "
                "have relevant coursework, a thesis, or certifications, consider listing them.")
    return ("The resume as a whole reads as less related to this posting than its "
            "individual sections. A summary line aimed at this kind of role may help.")


def seniority_note(posting_text: str):
    """
    Flag postings that ask for years of experience or a senior title.

    WHY: embeddings measure TOPIC overlap. A senior posting about the same
    topics as a junior resume looks similar to the model, so the score can't
    reflect seniority. We say so instead of pretending. This is a note, not a
    score penalty: parsing a candidate's years of experience from dates on a
    resume is too unreliable to penalise anyone with.
    """
    years = [int(n) for n in re.findall(r"(\d+)\s*\+?\s*years?", posting_text, re.I)]
    title = re.search(r"^Job Title:.*\b(senior|sr\.?|lead|principal|head|manager)\b",
                      posting_text, re.I | re.M)
    if not years and not title:
        return None
    parts = []
    if title:
        parts.append(f"is titled as a {title.group(1).lower()}-level role")
    if years:
        parts.append(f"mentions at least {max(years)} year(s) of experience")
    return ("This posting " + " and ".join(parts) + ". The score measures how closely "
            "your resume's content matches the posting's topics; it does not check "
            "seniority or years of experience, so weigh that separately.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def score_resume(model, resume: ParsedResume, posting_text: str) -> MatchResult:
    parts = split_posting(posting_text)
    notes = []
    if not parts["split_ok"]:
        notes.append("Couldn't find separate responsibilities/requirements sections in "
                     "the posting; compared against the whole posting instead.")

    seniority = seniority_note(posting_text)
    if seniority:
        notes.append(seniority)

    skills_c, matched, missing, extra = skills_component(resume.text, posting_text)

    components = {
        "skills": skills_c,
        "experience": semantic_component(
            model, "experience", resume.sections.get("EXPERIENCE", ""),
            parts["responsibilities"],
            "No Experience/Projects section was detected in the resume, so this "
            "part wasn't scored (rather than being counted as zero)."),
        "education": semantic_component(
            model, "education",
            degree_field(resume_degree_lines(resume.sections.get("EDUCATION", ""))),
            degree_field(parts["education"]),
            "Not scored: no Education section was detected in the resume "
            "(this is NOT counted as zero)." if not resume.sections.get("EDUCATION")
            else "Not scored: the posting doesn't mention education requirements."),
        "overall": semantic_component(
            model, "overall", resume.text, posting_text,
            "Resume or posting is empty."),
    }

    # Redistribute weight from components we couldn't compute, proportionally.
    # WHY: "we couldn't find your Education heading" must not silently become
    # "your education scored 0".
    total_available = sum(c.weight for c in components.values() if c.available)
    for c in components.values():
        c.effective_weight = c.weight / total_available if c.available and total_available else 0.0
        if not c.available:
            notes.append(f"{COMPONENT_LABELS[c.name]}: {c.note}")

    score = sum(c.score * c.effective_weight for c in components.values())

    skill_words = {w for s in extract_skills(posting_text) for w in s.lower().split()}
    keywords = missing_keywords(resume.text, posting_text, skill_words)

    low_confidence = not resume.found or resume.looks_scanned
    if low_confidence:
        notes.append("Low confidence: no standard resume sections were detected. "
                     "This may not be a resume, or its layout couldn't be parsed.")

    recs = build_recommendations(components, missing, extra, keywords, resume)
    return MatchResult(
        score=round(score, 1), components=components, matched_skills=matched,
        missing_skills=missing, extra_skills=extra, missing_keywords=keywords,
        recommendations=recs, low_confidence=low_confidence, notes=notes,
    )
