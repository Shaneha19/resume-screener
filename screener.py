"""Core engine for the AI Resume Screening System.

Pipeline: extract text -> clean -> extract skills/experience/education
-> score against Job Description -> rank candidates.
"""
import io
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    # languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust", "php", "ruby", "kotlin", "swift", "r", "sql",
    # web
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring boot", "rest api", "graphql",
    # data / ML
    "machine learning", "deep learning", "nlp", "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "data analysis", "data visualization", "power bi", "tableau", "excel", "statistics", "opencv", "transformers", "llm",
    # db / cloud / devops
    "mysql", "postgresql", "mongodb", "redis", "firebase", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "linux",
    "ci/cd", "jenkins", "terraform",
    # general
    "agile", "scrum", "oop", "data structures", "algorithms", "problem solving", "communication", "leadership", "teamwork",
]

EDU_LEVELS = {
    4: [r"ph\.?d", r"doctorate"],
    3: [r"m\.?tech", r"m\.?sc", r"\bmca\b", r"\bmba\b", r"master"],
    2: [r"b\.?tech", r"b\.?e\b", r"b\.?sc", r"\bbca\b", r"bachelor"],
    1: [r"diploma"],
}


@dataclass
class Result:
    name: str
    final_score: float = 0.0
    text_similarity: float = 0.0
    skill_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    years_experience: float = 0.0
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    extra_skills: list = field(default_factory=list)


# ---------- text extraction ----------
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- feature extraction ----------
def extract_skills(text: str) -> set:
    t = clean_text(text)
    found = set()
    for s in SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(s) + r"(?![a-z0-9+#])"
        if re.search(pattern, t):
            found.add(s)
    return found


def extract_years(text: str) -> float:
    t = text.lower()
    nums = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", t)]
    return max(nums) if nums else 0.0


def extract_edu_level(text: str) -> int:
    t = text.lower()
    for level in sorted(EDU_LEVELS, reverse=True):
        if any(re.search(p, t) for p in EDU_LEVELS[level]):
            return level
    return 0


# ---------- scoring ----------
def screen(jd_text: str, resumes: dict, weights=None) -> list:
    """resumes: {candidate_name: raw_text}. Returns list[Result] sorted best-first."""
    w = weights or {"text": 0.35, "skills": 0.40, "experience": 0.15, "education": 0.10}
    total = sum(w.values()) or 1
    w = {k: v / total for k, v in w.items()}

    jd_clean = clean_text(jd_text)
    jd_skills = extract_skills(jd_text)
    jd_years = extract_years(jd_text)
    jd_edu = extract_edu_level(jd_text)

    names = list(resumes)
    cleaned = [clean_text(resumes[n]) for n in names]
    if not names:
        return []

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform([jd_clean] + cleaned)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    results = []
    for name, sim in zip(names, sims):
        raw = resumes[name]
        r_sk = extract_skills(raw)
        matched = sorted(jd_skills & r_sk)
        missing = sorted(jd_skills - r_sk)
        extra = sorted(r_sk - jd_skills)

        skill_score = len(matched) / len(jd_skills) if jd_skills else 0.0
        years = extract_years(raw)
        exp_score = min(years / jd_years, 1.0) if jd_years else (1.0 if years else 0.5)
        edu = extract_edu_level(raw)
        edu_score = min(edu / jd_edu, 1.0) if jd_edu else (1.0 if edu else 0.5)
        text_score = min(sim * 2, 1.0)  # TF-IDF similarities are small; rescale

        final = (w["text"] * text_score + w["skills"] * skill_score
                 + w["experience"] * exp_score + w["education"] * edu_score)
        results.append(Result(
            name=name, final_score=round(final * 100, 1),
            text_similarity=round(text_score * 100, 1), skill_score=round(skill_score * 100, 1),
            experience_score=round(exp_score * 100, 1), education_score=round(edu_score * 100, 1),
            years_experience=years, matched_skills=matched, missing_skills=missing, extra_skills=extra,
        ))
    return sorted(results, key=lambda r: r.final_score, reverse=True)


def verdict(score: float) -> str:
    if score >= 70:
        return "Strong match - Shortlist"
    if score >= 50:
        return "Moderate match - Review"
    return "Weak match - Reject"
