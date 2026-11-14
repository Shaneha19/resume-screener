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
