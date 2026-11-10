"""
PDF -> text -> sections.

Library choice: pdfplumber (not pypdf).
WHY: on a real two-column-ish resume, pypdf returned text in the PDF's
internal drawing order. Headings got separated from their content (EDUCATION
and SKILLS headings appeared back-to-back, with the skills listed under
EDUCATION) and dates got attached to the wrong jobs. pdfplumber groups
characters by their position on the page, so it reads top-to-bottom,
left-to-right the way a human does, and kept every heading next to its
content. See the Phase 1 notes in docs/scoring_design.md.

Honest caveat: pdfplumber is better, not perfect. True multi-column layouts
(two independent text columns side by side) can still come out interleaved,
because it reads each horizontal line straight across the page.
"""

import re
from dataclasses import dataclass, field

import pdfplumber

from src.config import MAX_HEADING_WORDS, MIN_TEXT_CHARS, SECTION_HEADINGS

SECTION_NAMES = ["EXPERIENCE", "EDUCATION", "SKILLS", "OTHER"]


@dataclass
class ParsedResume:
    """Everything the rest of the pipeline needs from one resume."""

    text: str                                     # full cleaned text
    sections: dict = field(default_factory=dict)  # name -> text ("" if missing)
    found: list = field(default_factory=list)     # sections we DID detect
    missing: list = field(default_factory=list)   # sections we did NOT detect
    warnings: list = field(default_factory=list)  # mangled-text warnings

    @property
    def looks_scanned(self) -> bool:
        # Very little text from a PDF usually means it's an image (a scan or
        # an "export as picture" resume). Text extraction can't read images;
        # that needs OCR, which this tool doesn't do.
        return len(self.text.strip()) < MIN_TEXT_CHARS


# ---------------------------------------------------------------------------
# Step 1: PDF -> raw text
# ---------------------------------------------------------------------------
def extract_text(pdf_source) -> str:
    """
    Pull text out of a PDF.

    `pdf_source` may be a file path OR a file-like object (e.g. the bytes
    Streamlit gives us for an upload). Accepting file-like objects matters
    for privacy: the app can parse an upload entirely in memory without ever
    writing it to disk.
    """
    pages = []
    with pdfplumber.open(pdf_source) as pdf:
        for page in pdf.pages:
            gutter = _find_column_gutter(page)
            if gutter:
                # Two-column layout: read the whole left column, then the
                # whole right column, instead of straight across the page.
                left = page.crop((0, 0, gutter, page.height)).extract_text() or ""
                right = page.crop((gutter, 0, page.width, page.height)).extract_text() or ""
                pages.append(left + "\n" + right)
            else:
                # extract_text() returns None for pages with no text layer
                # (e.g. a scanned page), so fall back to "".
                pages.append(page.extract_text() or "")
    return clean_text("\n".join(pages))


def _find_column_gutter(page):
    """
    Detect a two-column layout. Returns the x position of the gap between
    the columns, or None for a normal single-column page.

    WHY: pdfplumber reads each horizontal line straight across the page. On
    a two-column resume that interleaves the columns ("SKILLS EXPERIENCE",
    "Python, SQL Data Analyst Intern...") and section detection fails. We
    verified this on a synthetic two-column PDF.

    Method: a gutter is a vertical line in the middle 60% of the page that
    NO word crosses, with a meaningful amount of text on both sides. On a
    single-column resume, full-width lines cross every candidate position,
    so no gutter is found. Limitation: three-column or mixed layouts
    (full-width header above two columns) are not handled.
    """
    words = page.extract_words()
    if len(words) < 20:
        return None
    best = None
    for x in range(int(page.width * 0.2), int(page.width * 0.8), 2):
        if any(w["x0"] < x < w["x1"] for w in words):
            continue  # some word crosses this line, so it's not a gutter
        left = sum(1 for w in words if w["x1"] <= x)
        right = len(words) - left
        # Both sides need at least 20% of the words, else it's just a
        # right-aligned date column or a margin note.
        if min(left, right) >= 0.2 * len(words):
            # Prefer the position with the most balanced split.
            balance = min(left, right)
            if best is None or balance > best[1]:
                best = (x, balance)
    return best[0] if best else None


def clean_text(text: str) -> str:
    """Normalise characters that cause matching problems later."""
    # Curly quotes / dashes -> plain ASCII, so "Bachelor's" matches "bachelor's".
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "-", " ": " ",
        "": "-",  # a common private-use bullet glyph in Word exports
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Collapse runs of spaces/tabs but KEEP newlines: line structure is how
    # we find headings.
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Step 2: text -> sections
# ---------------------------------------------------------------------------
# Matches a date range at the END of a line, e.g. "2022 - Present", "2024-2026".
# Many resumes put dates on the same line as the heading ("PROJECTS 2022 - Present"),
# so we strip them before deciding whether a line is a heading.
_TRAILING_DATES = re.compile(
    r"\s*((19|20)\d{2}|present|current)(\s*-\s*((19|20)\d{2}|present|current))?\s*$",
    re.IGNORECASE,
)

# Reverse lookup: "technical skills" -> "SKILLS", etc.
_HEADING_LOOKUP = {
    phrase: section
    for section, phrases in SECTION_HEADINGS.items()
    for phrase in phrases
}


def heading_section(line: str):
    """
    Return the section name if `line` looks like a heading, else None.

    The rule is deliberately strict: the WHOLE line (minus trailing dates and
    a trailing colon) must equal a known heading phrase. A looser "starts
    with" rule would misfire on content lines like
    "Languages: Python, SQL, Java" (which lives INSIDE the skills section)
    or "Experience using R and SQL" (a sentence, not a heading).
    """
    candidate = _TRAILING_DATES.sub("", line).strip()
    candidate = candidate.rstrip(":").strip().lower()
    if not candidate or len(candidate.split()) > MAX_HEADING_WORDS:
        return None
    return _HEADING_LOOKUP.get(candidate)


def detect_sections(text: str):
    """
    Split resume text into EXPERIENCE / EDUCATION / SKILLS / OTHER.

    Returns (sections, found, missing):
      sections: dict of section name -> text (always has all four keys,
                with "" for anything not detected, so callers never KeyError)
      found:    list of sections whose heading we actually saw
      missing:  list of sections we did not see

    Anything before the first recognised heading (name, contact details)
    goes into OTHER. If the same section appears twice (e.g. "Experience"
    and "Projects" both map to EXPERIENCE), the pieces are joined.
    """
    buckets = {name: [] for name in SECTION_NAMES}
    found = set()
    current = "OTHER"  # until we see a heading

    for line in text.splitlines():
        section = heading_section(line)
        if section:
            current = section
            found.add(section)
            continue  # don't include the heading word itself in the content
        buckets[current].append(line)

    sections = {name: "\n".join(lines).strip() for name, lines in buckets.items()}
    # OTHER always "exists" as the catch-all, so only report on the three
    # sections that the scorer actually depends on.
    core = ["EXPERIENCE", "EDUCATION", "SKILLS"]
    found_list = [s for s in core if s in found]
    missing_list = [s for s in core if s not in found]
    return sections, found_list, missing_list


# ---------------------------------------------------------------------------
# Step 3: sanity-check the text for extraction damage
# ---------------------------------------------------------------------------
def find_mangled_text(text: str) -> list:
    """
    Heuristics for text that extraction probably broke. Returns a list of
    human-readable warnings (empty list = nothing suspicious found).

    These are hints, not proof: a warning means "go look at this line".
    """
    warnings = []
    for line in text.splitlines():
        # (cid:123) is what PDF libraries print when a font has no mapping
        # from glyphs back to letters. The text is effectively unreadable.
        if "(cid:" in line:
            warnings.append(f"Unmapped font glyphs: {line[:80]!r}")
        # The Unicode replacement character means a byte couldn't be decoded.
        if "�" in line:
            warnings.append(f"Undecodable character: {line[:80]!r}")
        # Same letter 4+ times in a row ("Ssssskills") often means overlapping
        # text boxes or bold-by-overprinting got extracted multiple times.
        if re.search(r"([A-Za-z])\1{3,}", line):
            warnings.append(f"Repeated characters: {line[:80]!r}")
        # A very long run of letters with no spaces means spaces were lost
        # ("ExperiencedPythondeveloperwithstrong...").
        for token in re.findall(r"[A-Za-z]{25,}", line):
            warnings.append(f"Possible missing spaces: {token[:60]!r}")
        # Lots of single letters separated by spaces ("P y t h o n") means
        # letter-spaced text was extracted character by character.
        tokens = line.split()
        singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
        if len(tokens) >= 6 and singles / len(tokens) > 0.5:
            warnings.append(f"Letter-spaced text: {line[:80]!r}")
    return warnings


# ---------------------------------------------------------------------------
# Convenience wrapper used by the scorer and the app
# ---------------------------------------------------------------------------
def parse_resume(pdf_source) -> ParsedResume:
    """PDF (path or file-like) -> ParsedResume."""
    text = extract_text(pdf_source)
    return parse_resume_text(text)


def parse_resume_text(text: str) -> ParsedResume:
    """Same as parse_resume, for text you already have (used in tests)."""
    text = clean_text(text)
    sections, found, missing = detect_sections(text)
    return ParsedResume(
        text=text,
        sections=sections,
        found=found,
        missing=missing,
        warnings=find_mangled_text(text),
    )
