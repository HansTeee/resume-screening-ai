"""
Tests for skill extraction, focused on the bugs that word-boundary mistakes
cause. Run with:   python test_skills.py

Each case is (text, skills that MUST be found, skills that must NOT be found).
"""

from src.skills import extract_skills, count_skills, vocabulary_size

CASES = [
    # --- "Go" must not match inside other words or as the English verb ---
    ("Worked at Google on Go microservices", {"Go", "Microservices"}, set()),
    ("I worked at Google and used Gmail", set(), {"Go"}),
    ("We go above and beyond", set(), {"Go"}),
    ("Built a go-to-market dashboard", set(), {"Go"}),
    ("Backend written in Golang", {"Go"}, set()),
    ("Languages: Go, Python", {"Go", "Python"}, set()),

    # --- "R" must not match every word containing r, or R&D ---
    ("Experience with R, Python, SQL", {"R", "Python", "SQL"}, set()),
    ("Strong programmer, great learner", set(), {"R"}),
    ("Worked in the R&D department", set(), {"R"}),
    ("rare r characters everywhere", set(), {"R"}),

    # --- "C" / "C++" / "C#" ---
    ("Juan C. Cruz, programmer", set(), set()),  # documented weakness: see note below
    ("Proficient in C++ and C#", {"C++", "C#"}, {"C"}),
    ("Embedded C programming", {"C"}, set()),
    ("Reported to the C-suite", set(), {"C"}),

    # --- Java must not match JavaScript, and vice versa ---
    ("Java backend developer", {"Java"}, {"JavaScript"}),
    ("JavaScript and TypeScript", {"JavaScript", "TypeScript"}, {"Java"}),

    # --- aliases map to one canonical name ---
    ("JS, sklearn, TF", {"JavaScript", "scikit-learn", "TensorFlow"}, set()),
    ("scikit-learn pipelines", {"scikit-learn"}, set()),
    ("TF-IDF features", set(), {"TensorFlow"}),          # TF inside TF-IDF is not TensorFlow
    ("React.js frontend", {"React"}, set()),
    ("react quickly to incidents", set(), {"React"}),  # the verb, not the library

    # --- case insensitivity for unambiguous names ---
    ("PYTHON, pandas, NUMPY, Docker", {"Python", "pandas", "NumPy", "Docker"}, set()),

    # --- longer skill claims the text first ---
    ("Strong SQL & PL/SQL skills", {"SQL", "PL/SQL"}, set()),
    ("Built apps in React Native", {"React Native"}, {"React"}),
    ("Used MySQL and PostgreSQL", {"MySQL", "PostgreSQL"}, {"SQL"}),

    # --- ML / AI must not match inside other words ---
    ("HTML and CSS", {"HTML", "CSS"}, {"Machine Learning"}),
    ("Aimed to maintain systems", set(), {"Artificial Intelligence"}),
    ("AI/ML engineer", {"Artificial Intelligence", "Machine Learning"}, set()),

    # --- common words that are also skills ---
    ("Rest of the team", set(), {"REST APIs"}),
    ("Built REST APIs in Django", {"REST APIs", "Django"}, set()),
    ("Excellent communicator", set(), {"Excel"}),
    ("Rust-proof coating", set(), {"Rust"}),  # lowercase/hyphen: not the language

    # --- multi-word skills split across a PDF line break ---
    ("Applied machine\nlearning to forecasting", {"Machine Learning", "Forecasting"}, set()),
]


def run():
    failures = 0
    for text, must_have, must_not in CASES:
        found = set(extract_skills(text))
        missing = must_have - found
        wrong = must_not & found
        status = "PASS" if not missing and not wrong else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] {text!r:50} -> {sorted(found)}")
        if missing:
            print(f"        expected but not found: {sorted(missing)}")
        if wrong:
            print(f"        found but should not be: {sorted(wrong)}")

    # Counting: prominence ranking in Phase 2 depends on this.
    counts = count_skills("Docker, Docker, docker and Kubernetes")
    assert counts["Docker"] == 3 and counts["Kubernetes"] == 1, counts
    print("[PASS] counting: Docker x3, Kubernetes x1")

    print(f"\nVocabulary size: {vocabulary_size()} canonical skills")
    print(f"{len(CASES) - failures}/{len(CASES)} cases passed")
    # Known weakness, kept visible on purpose: a capital "C." initial
    # (as in "Juan C. Cruz") WILL be read as the C language. Case-sensitive
    # matching can't tell an initial from a language name.
    print("Known weakness:", repr("Juan C. Cruz"), "->", extract_skills("Juan C. Cruz"))
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
