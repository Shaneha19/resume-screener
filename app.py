import pandas as pd
import streamlit as st

from screener import extract_text, screen, verdict

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption("Upload a job description and resumes. The system ranks candidates using TF-IDF similarity, skill matching, experience and education.")

with st.sidebar:
    st.header("Scoring weights")
    w_text = st.slider("Text similarity", 0, 100, 35)
    w_skill = st.slider("Skill match", 0, 100, 40)
    w_exp = st.slider("Experience", 0, 100, 15)
    w_edu = st.slider("Education", 0, 100, 10)
    shortlist_cutoff = st.slider("Shortlist cutoff (%)", 0, 100, 70)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload JD (txt/pdf/docx) or paste below", type=["txt", "pdf", "docx"], key="jd")
    jd_text = st.text_area("Job description text", height=220)
    if jd_file:
        jd_text = extract_text(jd_file.name, jd_file.read())
with col2:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload resumes (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("🔍 Screen candidates", type="primary"):
    if not jd_text.strip() or not files:
        st.error("Please provide a job description and at least one resume.")
        st.stop()
    resumes = {f.name: extract_text(f.name, f.read()) for f in files}
    weights = {"text": w_text, "skills": w_skill, "experience": w_exp, "education": w_edu}
    results = screen(jd_text, resumes, weights)

    df = pd.DataFrame([{
        "Rank": i + 1, "Candidate": r.name, "Final Score": r.final_score,
        "Text Sim": r.text_similarity, "Skills": r.skill_score,
        "Experience": r.experience_score, "Education": r.education_score,
        "Years": r.years_experience, "Verdict": verdict(r.final_score),
    } for i, r in enumerate(results)])

    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Candidate")["Final Score"])
    st.success(f"{(df['Final Score'] >= shortlist_cutoff).sum()} candidate(s) above the {shortlist_cutoff}% cutoff.")
    st.download_button("⬇ Download CSV", df.to_csv(index=False), "screening_results.csv", "text/csv")

    st.subheader("Candidate details")
    for r in results:
        with st.expander(f"{r.name} - {r.final_score}% - {verdict(r.final_score)}"):
            st.write("✅ **Matched skills:**", ", ".join(r.matched_skills) or "None")
            st.write("❌ **Missing skills:**", ", ".join(r.missing_skills) or "None")
            st.write("➕ **Additional skills:**", ", ".join(r.extra_skills) or "None")
