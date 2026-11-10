"""
Skill vocabulary and extraction.

The vocabulary is hand-curated for tech / ML / data roles. It will NOT work
well for nursing, accounting, law, etc. - that is a known limitation, and it
is also why an unrelated-field posting gets very few skills extracted.

Format: canonical name -> list of aliases. The canonical name is what we
display; any alias (or the canonical name itself) counts as a match.
"""

import re
from collections import Counter

SKILL_VOCABULARY = {
    "Programming languages": {
        "Python": ["python", "python3"],
        "Java": ["java"],
        "JavaScript": ["javascript", "JS", "ecmascript"],
        "TypeScript": ["typescript", "TS"],
        "C": ["C"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "csharp"],
        "Go": ["Go", "golang"],
        "Rust": ["Rust"],
        "R": ["R"],
        "SQL": ["sql"],
        "PL/SQL": ["pl/sql", "plsql"],
        "Scala": ["scala"],
        "Kotlin": ["kotlin"],
        "Swift": ["Swift"],
        "Ruby": ["Ruby"],
        "PHP": ["php"],
        "MATLAB": ["matlab"],
        "Bash": ["bash", "shell scripting"],
        "PowerShell": ["powershell"],
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3"],
        "Dart": ["Dart"],
        "Verilog": ["verilog"],
        "VHDL": ["vhdl"],
        "Assembly": ["assembly language"],
        "Julia": ["Julia"],
        "Perl": ["perl"],
        "Lua": ["lua"],
        "Haskell": ["haskell"],
        "Solidity": ["solidity"],
    },
    "Web & backend frameworks": {
        "React": ["React", "react.js", "reactjs"],
        "React Native": ["react native"],
        "Angular": ["angular", "angularjs"],
        "Vue.js": ["vue", "vue.js", "vuejs"],
        "Next.js": ["next.js", "nextjs"],
        "Svelte": ["svelte"],
        "Node.js": ["node.js", "nodejs"],
        "Express.js": ["express.js", "expressjs"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Spring Boot": ["spring boot", "springboot"],
        "Spring": ["Spring"],
        "Ruby on Rails": ["ruby on rails", "Rails"],
        "Laravel": ["laravel"],
        ".NET": [".net", "dotnet", "asp.net"],
        "REST APIs": ["rest api", "rest apis", "restful", "restful api", "restful apis", "REST"],
        "GraphQL": ["graphql"],
        "gRPC": ["grpc"],
        "Microservices": ["microservices", "microservice"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "Bootstrap": ["bootstrap"],
        "jQuery": ["jquery"],
        "Flutter": ["flutter"],
        "Streamlit": ["streamlit"],
        "Gradio": ["gradio"],
    },
    "ML & data libraries": {
        "NumPy": ["numpy"],
        "pandas": ["pandas"],
        "Polars": ["polars"],
        "scikit-learn": ["scikit-learn", "scikit learn", "sklearn", "scikitlearn"],
        "TensorFlow": ["tensorflow", "TF", "tensorflow 2"],
        "Keras": ["keras"],
        "PyTorch": ["pytorch", "torch"],
        "JAX": ["jax"],
        "XGBoost": ["xgboost"],
        "LightGBM": ["lightgbm"],
        "CatBoost": ["catboost"],
        "Hugging Face": ["hugging face", "huggingface", "hf transformers"],
        "sentence-transformers": ["sentence-transformers", "sentence transformers", "sbert"],
        "spaCy": ["spacy"],
        "NLTK": ["nltk"],
        "OpenCV": ["opencv", "cv2"],
        "Matplotlib": ["matplotlib"],
        "Seaborn": ["seaborn"],
        "Plotly": ["plotly"],
        "SciPy": ["scipy"],
        "statsmodels": ["statsmodels"],
        "Jupyter": ["jupyter", "jupyter notebook", "jupyter notebooks", "jupyterlab"],
        "LangChain": ["langchain"],
        "LlamaIndex": ["llamaindex", "llama index"],
        "MLflow": ["mlflow"],
        "Kubeflow": ["kubeflow"],
        "Weights & Biases": ["weights & biases", "weights and biases", "wandb"],
        "ONNX": ["onnx"],
        "Apache Spark": ["spark", "apache spark", "pyspark"],
        "Hadoop": ["hadoop"],
        "Apache Kafka": ["kafka", "apache kafka"],
        "Apache Airflow": ["airflow", "apache airflow"],
        "dbt": ["dbt"],
        "Dask": ["dask"],
        "SMOTE": ["smote"],
    },
    "ML & data concepts": {
        "Machine Learning": ["machine learning", "ML"],
        "Deep Learning": ["deep learning", "deep learning models"],
        "Artificial Intelligence": ["artificial intelligence", "AI"],
        "Neural Networks": ["neural networks", "neural network", "artificial neural networks", "ANN"],
        "Supervised Learning": ["supervised learning"],
        "Unsupervised Learning": ["unsupervised learning"],
        "Reinforcement Learning": ["reinforcement learning"],
        "Computer Vision": ["computer vision"],
        "NLP": ["nlp", "natural language processing"],
        "LLMs": ["llm", "llms", "large language models", "large language model"],
        "RAG": ["RAG", "retrieval-augmented generation", "retrieval augmented generation"],
        "Prompt Engineering": ["prompt engineering"],
        "Generative AI": ["generative ai", "genai", "gen ai"],
        "Transformers (architecture)": ["transformer models", "transformer architecture"],
        "CNNs": ["cnn", "cnns", "convolutional neural network", "convolutional neural networks"],
        "RNNs": ["rnn", "rnns", "recurrent neural network", "recurrent neural networks"],
        "LSTM": ["lstm", "lstms"],
        "Regression": ["regression", "linear regression", "logistic regression"],
        "Classification": ["classification", "classifier", "classifiers"],
        "Clustering": ["clustering", "k-means", "kmeans"],
        "Decision Trees": ["decision tree", "decision trees", "decision tree learning"],
        "Random Forest": ["random forest", "random forests"],
        "SVM": ["svm", "svms", "support vector machine", "support vector machines"],
        "KNN": ["knn", "k-nearest neighbors", "k-nearest neighbours"],
        "Gradient Boosting": ["gradient boosting"],
        "Feature Engineering": ["feature engineering"],
        "Model Evaluation": ["model evaluation"],
        "Hyperparameter Tuning": ["hyperparameter tuning", "hyperparameter optimization"],
        "Anomaly Detection": ["anomaly detection", "detect anomalies"],
        "Forecasting": ["forecasting"],
        "Time Series": ["time series", "time-series"],
        "Recommender Systems": ["recommender systems", "recommendation systems", "recommender system"],
        "Speech / Sound Processing": ["sound processing", "speech recognition", "audio processing"],
        "Data Drift Monitoring": ["data drift", "concept drift", "model drift"],
        "Model Deployment": ["model deployment", "deploying models", "models deployed to production"],
        "MLOps": ["mlops"],
        "A/B Testing": ["a/b testing", "ab testing", "a/b tests"],
        "Statistics": ["statistics", "statistical", "statistical analysis", "statistical tests"],
        "Probability": ["probability"],
        "Linear Algebra": ["linear algebra"],
        "Calculus": ["calculus"],
        "Algorithms": ["algorithms", "analysis of algorithms", "data structures and algorithms"],
        "Data Structures": ["data structures"],
        "Data Analysis": ["data analysis", "exploratory data analysis", "eda", "exploratory analysis"],
        "Data Visualization": ["data visualization", "data visualisation"],
        "Data Engineering": ["data engineering"],
        "Data Pipelines": ["data pipelines", "data pipeline", "etl", "elt"],
        "Data Warehousing": ["data warehouse", "data warehousing"],
        "Data Lake / Lakehouse": ["data lake", "lakehouse", "data lakehouse"],
        "Data Architecture": ["data architecture", "data architectures"],
        "Big Data": ["big data"],
        "Backtesting": ["backtesting"],
        "Data Science": ["data science", "data scientist", "data scientists"],
        "Software Engineering": ["software engineering", "software development"],
    },
    "Databases": {
        "PostgreSQL": ["postgresql", "postgres"],
        "MySQL": ["mysql"],
        "SQLite": ["sqlite"],
        "Microsoft SQL Server": ["sql server", "mssql"],
        "Oracle Database": ["oracle database", "oracle db"],
        "MongoDB": ["mongodb", "mongo"],
        "Redis": ["redis"],
        "Cassandra": ["cassandra"],
        "Elasticsearch": ["elasticsearch"],
        "DynamoDB": ["dynamodb"],
        "Firebase": ["firebase", "firestore"],
        "Supabase": ["supabase"],
        "Neo4j": ["neo4j"],
        "Vector Databases": ["vector database", "vector databases", "pinecone", "faiss", "chromadb", "weaviate"],
        "Snowflake": ["snowflake"],
        "Databricks": ["databricks"],
        "BigQuery": ["bigquery"],
        "Redshift": ["redshift"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure", "microsoft azure"],
        "GCP": ["gcp", "google cloud", "google cloud platform"],
        "Alibaba Cloud": ["alibaba cloud", "alibaba"],
        "AWS SageMaker": ["sagemaker"],
        "AWS Lambda": ["aws lambda"],
        "AWS S3": ["s3"],
        "AWS EC2": ["ec2"],
        "Azure ML": ["azure ml", "azure machine learning"],
        "Vertex AI": ["vertex ai"],
        "Cloud Computing": ["cloud computing", "cloud services", "cloud platforms", "cloud platform"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Containerization": ["containerization", "containerisation", "containers"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
        "GitHub Actions": ["github actions"],
        "Jenkins": ["jenkins"],
        "GitLab CI": ["gitlab ci"],
        "Linux": ["linux", "ubuntu", "unix"],
        "Nginx": ["nginx"],
        "Heroku": ["heroku"],
        "Vercel": ["vercel"],
        "Prometheus": ["prometheus"],
        "Grafana": ["grafana"],
        "Monitoring": ["monitoring", "observability"],
        "Serverless": ["serverless"],
    },
    "Tools": {
        "Git": ["git"],
        "GitHub": ["github"],
        "GitLab": ["gitlab"],
        "Bitbucket": ["bitbucket"],
        "Jira": ["jira"],
        "Confluence": ["confluence"],
        "Postman": ["postman"],
        "VS Code": ["vs code", "vscode", "visual studio code"],
        "Excel": ["Excel", "microsoft excel", "ms excel"],
        "Google Sheets": ["google sheets"],
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau"],
        "Looker": ["looker"],
        "Figma": ["figma"],
        "Arduino": ["arduino"],
        "Raspberry Pi": ["raspberry pi"],
        "IoT": ["iot", "internet of things", "internet-of-things"],
        "MetaTrader": ["metatrader", "mt5", "metatrader 5"],
        "Kaggle": ["kaggle"],
        "LaTeX": ["latex"],
        "Unit Testing": ["unit testing", "unit tests", "pytest", "junit"],
        "Agile": ["agile", "scrum", "kanban"],
        "OOP": ["oop", "object-oriented programming", "object oriented programming"],
        "System Design": ["system design", "system architecture"],
        "Full-Stack Development": ["full-stack", "full stack", "fullstack"],
        "Frontend Development": ["frontend", "front-end", "front end"],
        "Backend Development": ["backend", "back-end", "back end"],
        "Cybersecurity": ["cybersecurity", "cyber security", "information security"],
        "Networking": ["computer networking", "tcp/ip"],
        "Embedded Systems": ["embedded systems", "embedded system", "microcontrollers"],
        "Digital Logic Design": ["digital logic design", "digital logic"],
    },
    "Soft skills": {
        "Communication": ["communication", "communication skills"],
        "Leadership": ["leadership", "leading a team", "team lead", "led a team"],
        "Teamwork": ["teamwork", "team player", "collaboration", "collaborative"],
        "Problem Solving": ["problem solving", "problem-solving"],
        "Critical Thinking": ["critical thinking"],
        "Stakeholder Management": ["stakeholder management", "stakeholders"],
        "Mentoring": ["mentoring", "mentorship", "mentored"],
        "Time Management": ["time management"],
        "Presentation Skills": ["presentation skills", "public speaking"],
        "Project Management": ["project management"],
        "Technical Writing": ["technical writing", "documentation", "document source code"],
        "Adaptability": ["adaptability", "fast-paced environment", "fast-paced"],
        "Attention to Detail": ["attention to detail", "detail-oriented"],
    },
}

# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------
# Most aliases match case-insensitively ("python" == "Python" == "PYTHON").
# But some aliases are ALSO ordinary English words or initials, so matching
# them case-insensitively produces garbage:
#   "go"   -> "go to market", "we go above and beyond"
#   "r"    -> any stray letter r
#   "c"    -> a middle initial like "Juan C. Cruz"
#   "rust", "swift", "ruby", "spring", "excel", "julia", "dart" -> words/names
#   "ai", "ml", "js", "ts", "tf", "rag" -> short enough to collide
#   "rest", "rails", "react" -> "the rest of", "guardrails", "react quickly"
# An alias is matched case-sensitively when it's written in the vocabulary
# with at least one uppercase letter AND it's in this set. Writing it as
# "Go" means only "Go" counts, never "go".
CASE_SENSITIVE_ALIASES = {
    "Go", "R", "C", "Rust", "Swift", "Ruby", "Spring", "Excel", "Julia", "Dart",
    "AI", "ML", "JS", "TS", "TF", "RAG", "ANN", "REST", "Rails", "React",
}

# Characters allowed INSIDE a skill name but which count as "not a letter or
# digit" for boundary purposes. We define the boundary ourselves instead of
# using \b, because \b treats "+" and "#" as boundaries, so \bC++\b can't
# work, and \b also happily matches the "R" in "R&D".
_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9])"
# After the skill: not followed by a letter/digit. Also not followed by
# "&" + letter, so "R&D" doesn't count as the language R, and not by "-"
# + letter, so "C-suite" and "Go-to-market" don't count either.
_BOUNDARY_AFTER = r"(?![A-Za-z0-9])(?![&\-][A-Za-z])"


def _build_patterns():
    """
    Turn the vocabulary into a list of (compiled_regex, canonical_name).

    Sorted LONGEST alias first. WHY: we "consume" text as we match it, so
    "react native" gets claimed as React Native before the shorter "react"
    gets a chance to also claim it, and "pl/sql" is claimed before "sql".
    """
    entries = []
    for category in SKILL_VOCABULARY.values():
        for canonical, aliases in category.items():
            # The canonical name itself also counts as an alias, unless it's
            # ambiguous in a way the alias list already handles.
            for alias in set(aliases) | {canonical}:
                entries.append((alias, canonical))
    entries.sort(key=lambda e: len(e[0]), reverse=True)

    patterns = []
    for alias, canonical in entries:
        # re.escape so "c++", ".net", "ci/cd" are matched literally.
        # Spaces in the alias match any whitespace (PDF line breaks).
        body = re.escape(alias).replace(r"\ ", r"\s+")
        flags = 0 if alias in CASE_SENSITIVE_ALIASES else re.IGNORECASE
        regex = re.compile(_BOUNDARY_BEFORE + body + _BOUNDARY_AFTER, flags)
        patterns.append((regex, canonical))
    return patterns


_PATTERNS = _build_patterns()

# canonical -> category, for grouping in the UI
SKILL_CATEGORY = {
    canonical: category
    for category, skills in SKILL_VOCABULARY.items()
    for canonical in skills
}


def count_skills(text: str) -> Counter:
    """
    Count how many times each skill is mentioned.

    Returns Counter(canonical_name -> mentions). Counts matter for job
    postings: a skill mentioned three times is probably more important to
    the employer than one mentioned once.
    """
    counts = Counter()
    # We overwrite matched characters with spaces as we go ("consuming"
    # them), so a longer skill and a shorter skill can't both claim the same
    # characters. Using spaces keeps string positions stable.
    working = text
    for regex, canonical in _PATTERNS:
        def _consume(match):
            counts[canonical] += 1
            return " " * len(match.group(0))
        working = regex.sub(_consume, working)
    return counts


def extract_skills(text: str) -> list:
    """Return the sorted list of distinct canonical skills found in `text`."""
    return sorted(count_skills(text), key=str.lower)


def vocabulary_size() -> int:
    return sum(len(skills) for skills in SKILL_VOCABULARY.values())
