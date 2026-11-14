# AI Resume Screening System

A web app that ranks resumes against a job description using NLP and machine learning techniques.

## Features
- Upload resumes in PDF, DOCX or TXT (bulk upload) and a job description (file or pasted text)
- Text extraction and cleaning
- Skill extraction (70+ skills, exact-match so "java" does not match "javascript")
- Experience (years) and education-level detection
- Weighted score: TF-IDF cosine similarity + skill match + experience + education
- Ranking table, bar chart, matched / missing / extra skills per candidate
- Verdict (Shortlist / Review / Reject), adjustable weights, CSV export

## How it works
1. Extract text (pypdf / python-docx)
2. Clean text (lowercase, remove noise)
3. TF-IDF (1-2 grams) of JD + resumes, then cosine similarity
4. Regex-based skill / experience / education extraction
5. Final score = w1*text + w2*skills + w3*experience + w4*education (default 35/40/15/10)
6. Sort descending, show verdict

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Try it with the files in `sample_data/`. Run tests: `python -m pytest test_screener.py`

## Project structure
```
app.py            Streamlit UI
screener.py       Extraction + scoring engine
test_screener.py  Unit tests
sample_data/      Sample JD and resumes
```

## 45-Day Plan
| Days | Work |
|------|------|
| 1-5 | Problem study, literature on ATS / resume parsing, set up Python, Git, Streamlit |
| 6-12 | Text extraction from PDF/DOCX, text cleaning |
| 13-20 | TF-IDF, cosine similarity, skill dictionary and extraction |
| 21-27 | Experience / education extraction, weighted scoring |
| 28-35 | Streamlit UI, charts, CSV export |
| 36-40 | Testing, bug fixes, accuracy check on 20+ resumes |
| 41-45 | Documentation, report, PPT, demo / viva preparation |

## Limitations and future scope
- Scanned image PDFs need OCR (pytesseract)
- Skill list is manual; can be extended with spaCy NER or an LLM
- Upgrade similarity with sentence-transformers (BERT embeddings)
- Add database, login, bias-check (hide name/gender) and email integration

## Viva questions to prepare
- Why TF-IDF and cosine similarity? What is a 1-2 gram?
- Why not only keyword matching? (semantic gaps, keyword stuffing)
- How would you reduce bias? How would you evaluate accuracy (precision / recall vs HR ranking)?
