# Scoring design

This document was written **before** the scorer was coded, then updated
with measured numbers in Phase 2. Sections marked *(measured)* contain real
output from running on real files.

## What the score is (and isn't)

The score is an **estimate of how well one resume's text matches one job
posting's text**. It is not a measure of a person's ability, and it is not
a hiring prediction. It exists to give the *candidate* feedback on how
their resume reads against a posting they've chosen.

## The four components

| Component        | Weight | Method                                                   |
|------------------|-------:|----------------------------------------------------------|
| Skills match     | 40%    | Keyword matching against a curated vocabulary            |
| Experience match | 30%    | Embedding similarity: resume EXPERIENCE vs posting       |
| Education match  | 15%    | Embedding similarity: resume EDUCATION vs posting        |
| Overall semantic | 15%    | Embedding similarity: whole resume vs whole posting      |

Final score = weighted sum of the four component scores, each 0-100.

## The weights are judgement calls

**None of these weights were learned, fitted, or validated.** There is no
dataset here of "correct" resume/posting match scores to fit them against,
so any claim that 40/30/15/15 is *optimal* would be invented. What follows
is the reasoning for each choice. It is an argument you can disagree with,
not a derivation.

### Skills match - 40% (the largest weight)
- **It produces the most actionable feedback.** "This posting mentions
  Docker three times and your resume doesn't mention it" is something a
  candidate can act on today. A similarity of 0.62 is not.
- **It's closest to what ATS keyword filters actually do.** Many applicant
  tracking systems screen on literal keyword presence, so this component
  approximates a real hurdle candidates face.
- **Why not higher:** keyword matching is easy to game (paste a skills list)
  and blind to context ("used Docker once" and "ran Docker in production
  for 3 years" look the same). Letting it dominate further would reward
  keyword stuffing.

### Experience match - 30%
- Hiring for most roles cares more about what you've done than where you
  studied, so experience gets the second-largest share.
- It is semantic (embeddings), so it can credit related work described in
  different words ("built a forecasting model" vs "time-series prediction").
- **Why not higher:** embedding similarity measures *topic overlap*, not
  depth, seniority, or quality. A junior and a senior describing the same
  kind of project look very similar. That limitation makes me unwilling to
  let it outweigh the explicit skills check.

### Education match - 15%
- Many postings do list degree requirements, so it shouldn't be zero.
- Kept low deliberately: degree filtering is one of the more exclusionary
  parts of hiring, and education sections are short and formulaic, so their
  embeddings carry little information.

### Overall semantic - 15%
- A safety net. Section detection is heuristic and **will** fail on some
  layouts. When it does, the whole-resume comparison still captures fit.
- Kept low because it overlaps with the experience component; weighting it
  heavily would double-count the same signal.

## Hybrid design: why not pure embeddings?

Pure embedding similarity gives one opaque number. Keyword matching gives
explicit **matched** and **missing** lists, and those lists are what the
improvement advice is built from. Embeddings cover what keywords miss
(paraphrase, related domains). Each method covers the other's blind spot.

## Graceful degradation when sections are missing

Section detection fails often on real resumes (unusual headings, sidebars,
icons instead of words). Rules:

1. **Missing resume section → its weight is redistributed** proportionally
   across the components that *can* be computed. It is **not** scored as 0.
   A missing "Education" heading means *we couldn't find it*, not *the
   person has no education*, and scoring it 0 would silently punish people
   for their resume template.
2. **Skills come from the whole resume text**, not just the SKILLS section,
   so a missing skills heading doesn't hide skills listed elsewhere.
3. **No sections detected at all → low-confidence flag** shown to the user,
   and only skills + overall semantic are used.
4. **Posting lists no recognisable skills** (e.g. a non-tech posting) →
   skills match is reported as "not measurable" and its weight is
   redistributed. This is flagged, since it usually means the posting is
   outside the vocabulary's domain.

Every redistribution is reported in the output, so the user can see when a
score rests on fewer components than usual.

## Calibrating raw similarity *(measured)*

Cosine similarity between two pieces of English prose is rarely near 0,
because unrelated texts still share common words and sentence structure.
Showing raw cosine × 100 would compress every result into a narrow band.

Method: score the resume against postings from completely unrelated fields
(nursing, accounting). Whatever that scores is the effective **floor**.
Then rescale:

    displayed = clip((raw - FLOOR) / (CEILING - FLOOR), 0, 1) * 100

See the "Measured results" section below for the numbers.

## Measured results

All numbers below come from `compare_models.py` and `test_scorer.py`, run on
one real resume, four real Kalibrr postings, two synthetic unrelated
postings (nursing, accounting) for calibration, and one held-out synthetic
unrelated posting (Grade 4 teacher) for testing. That is a small sample:
treat the conclusions as directional.

### PDF extraction (Phase 1)

| Library    | Result on a real resume |
|------------|-------------------------|
| pypdf      | Text in internal drawing order. EDUCATION and SKILLS headings came out back-to-back with the skills list under EDUCATION; date ranges attached to the wrong jobs; summary paragraph at the very end. |
| pdfplumber | Correct top-to-bottom reading order; every heading next to its content; dates on the right lines. **Chosen.** |

### Token truncation

`all-MiniLM-L6-v2` truncates at 256 tokens, `all-mpnet-base-v2` at 384. The
real resume's experience section was 406 tokens and the full resume 611,
so without handling it MiniLM would have ignored 37% of the experience
section and 58% of the resume. `embed()` therefore splits text into
~120-word chunks, embeds each, and averages.

### Model choice

Timings with torch limited to 2 threads (a free CPU Space has 2 vCPU):

|                                        | all-MiniLM-L6-v2 | all-mpnet-base-v2 |
|----------------------------------------|-----------------:|------------------:|
| Dimensions                             | 384              | 768               |
| Size on disk                           | 92 MB            | 439 MB            |
| Time per analysis (1 resume, 1 posting)| ~0.2 s           | ~1.3 s            |

Raw cosine similarity, resume vs posting:

| Posting                    | MiniLM exp | MiniLM whole | mpnet exp | mpnet whole |
|----------------------------|-----------:|-------------:|----------:|------------:|
| Cloud Data Scientist / AI-ML | 0.305 | 0.505 | **0.568** | 0.521 |
| MLOps Specialist           | 0.369 | 0.424 | 0.469 | 0.519 |
| Senior Data Scientist      | 0.213 | 0.481 | 0.340 | 0.483 |
| Data Engineer (Databricks) | 0.138 | 0.403 | 0.177 | 0.369 |
| *Nursing (unrelated)*      | 0.061 | 0.140 | 0.043 | 0.123 |
| *Accounting (unrelated)*   | 0.061 | 0.300 | 0.085 | 0.203 |

**Chosen: all-mpnet-base-v2.** Both models rank unrelated postings lowest,
but mpnet separates them more clearly: the gap between the weakest
related posting and the worst-case unrelated one is 0.166 on the whole
resume (MiniLM: 0.103), and its experience scores spread over 0.04-0.57
rather than 0.06-0.37. Its experience ranking also puts the AI/ML posting
first, which fits an ML-focused resume; MiniLM put MLOps first. The cost is
~1.1 s more per analysis and 350 MB more to download, acceptable for a tool
that runs one analysis per click. The download is moved to build time
with `preload_from_hub` in the Space config.

### Education: a bug found by measuring

Comparing full education lines ranked *Bachelor of Science in Nursing*
**above** *Bachelor's degree in Math, Statistics, ... Engineering* for a
Computer Engineering graduate (mpnet 0.375 vs 0.267). Short degree lines
share one template and the template dominated. Fix: strip degree
boilerplate and institution names and compare only the field of study.
After the fix: Engineering-listing posting 0.445, nursing 0.228,
accounting 0.275.

### The similarity floor and calibration

Floors (raw cosine of the unrelated-field postings, mpnet):

| Component  | Nursing | Accounting | Floor used |
|------------|--------:|-----------:|-----------:|
| Experience | 0.043   | 0.085      | **0.085**  |
| Education  | 0.228   | 0.275      | **0.275**  |
| Overall    | 0.123   | 0.203      | **0.203**  |

Ceilings were measured with synthetic experience sections written to
paraphrase a posting's responsibilities (what a strongly tailored resume
scores): experience 0.785 / 0.708 → ceiling **0.75**; whole resume
0.737 / 0.706 → **0.72**. Education: a field the posting explicitly lists
scored 0.41-0.445 → **0.45**.

Without calibration, raw × 100 on the whole resume would have shown the
unrelated accounting posting at 20% and every real posting squeezed
between 37% and 52%: one narrow band, exactly the failure the design
warned about. (Under MiniLM it was worse: accounting at 30%.)

**Circularity check.** Calibrating on nursing and accounting makes *those*
score ~0 by construction, so they can't test the calibration. The test
uses a third unrelated posting held out from calibration (Grade 4
teacher).

### End-to-end results (test_scorer.py)

| Posting                         | Score | Skills | Exp  | Edu  | Overall |
|---------------------------------|------:|-------:|-----:|-----:|--------:|
| Cloud Data Scientist / AI-ML    | 54.0  | 21.1   | 72.7 | 96.9 | 61.5    |
| Senior Data Scientist           | 45.1  | 46.7   | 38.3 | n/a  | 54.2    |
| MLOps Specialist                | 40.0  | 18.8   | 57.8 | n/a  | 61.1    |
| Data Engineer (Databricks)      | 19.1  | 18.2   | 13.8 | n/a  | 32.0    |
| *Grade 4 Teacher (held out)*    | 3.8   | n/a    | 7.7  | 0.0  | 0.0     |

The unrelated posting is clearly lowest (15.3 points below the next).
**The senior role does NOT score lowest among the tech roles.** It
scores second, because its listed techniques (regression, decision trees,
neural networks, SQL, Python, statistics) genuinely appear in the resume.
This is the expected blind spot of a content-matching tool: it measures
topic overlap, not seniority. Rather than add an unvalidated seniority
penalty, the tool shows a note whenever a posting has a senior title or
asks for N+ years.
