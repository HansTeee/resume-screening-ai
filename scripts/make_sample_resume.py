"""
Generate the SYNTHETIC sample resume used by the "Try the sample" button.

Alex Rivera is an invented person: invented name, a reserved example.com
email, no phone number, no address, a fictional university and employer.
Never replace this with a real resume.

Also generates an "after" version used to test whether following the tool's
advice raises the score (README pair 6). The rule for "after": it may only
NAME things the "before" resume already DESCRIBES (e.g. "packaged it in a
container" -> "packaged it with Docker"). Nothing is invented. That mirrors
what the tool asks of real users.

Dev-only dependency:  pip install fpdf2
Usage:  python scripts/make_sample_resume.py
Writes: docs/sample_resume.pdf        (committed)
        data/sample_resume_after.pdf  (gitignored, test only)
"""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent

BEFORE = {
    "header": ["ALEX RIVERA", "alex.rivera@example.com"],
    "SUMMARY": [
        "Computer Science graduate who builds machine learning projects end to end, "
        "from data collection to a working demo.",
    ],
    "EDUCATION": [
        "BS Computer Science, Example State University, 2021-2025",
        "Thesis: Classifying customer support tickets with natural language processing",
    ],
    "EXPERIENCE": [
        "Machine Learning Intern - Fictional Retail Co., Jun 2024 - Aug 2024",
        "- Built a text classifier in PyTorch that routed customer support tickets into 12 categories",
        "- Wrapped the model in a Python web API and packaged it in a container for the deployment team",
        "- Wrote SQL queries to pull 80,000 labelled tickets from the company database",
        "- Added automated tests that ran on every pull request",
    ],
    "PROJECTS": [
        "Movie Review Sentiment Analyzer",
        "- Fine-tuned a pretrained language model on 50,000 reviews, reaching 91% accuracy",
        "- Deployed the demo on a cloud virtual machine so classmates could try it",
        "Bike Share Demand Forecast",
        "- Predicted hourly rentals from weather data using gradient boosting in scikit-learn",
    ],
    "SKILLS": [
        "Languages: Python, SQL",
        "Libraries: PyTorch, pandas, scikit-learn",
        "Tools: Git, Linux",
    ],
}

# Same person, same experience - only the WORDING follows the tool's advice.
AFTER = {
    **BEFORE,
    "EXPERIENCE": [
        "Machine Learning Intern - Fictional Retail Co., Jun 2024 - Aug 2024",
        "- Built an NLP text classifier in PyTorch that routed customer support tickets into 12 categories",
        "- Served the model as a FastAPI REST API and packaged it with Docker for the deployment team",
        "- Wrote SQL queries to pull 80,000 labelled tickets from the company database",
        "- Added unit tests that ran in a CI/CD pipeline on every pull request",
    ],
    "PROJECTS": [
        "Movie Review Sentiment Analyzer",
        "- Fine-tuned a pretrained transformer model (NLP) on 50,000 reviews, reaching 91% accuracy",
        "- Deployed the demo with Docker on an AWS EC2 virtual machine",
        "Bike Share Demand Forecast",
        "- Predicted hourly rentals from weather data using gradient boosting in scikit-learn",
    ],
    "SKILLS": [
        "Languages: Python, SQL",
        "Libraries: PyTorch, pandas, scikit-learn",
        "Tools: Git, Linux, Docker, FastAPI, AWS (EC2), CI/CD",
    ],
}


def build(content: dict, path: Path):
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    name, email = content["header"]
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 9, name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, email, new_x="LMARGIN", new_y="NEXT")
    for section in ["SUMMARY", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, section, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in content[section]:
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    path.parent.mkdir(exist_ok=True)
    pdf.output(str(path))
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    build(BEFORE, ROOT / "docs" / "sample_resume.pdf")
    build(AFTER, ROOT / "data" / "sample_resume_after.pdf")
