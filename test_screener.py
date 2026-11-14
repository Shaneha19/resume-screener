from screener import screen, extract_skills, extract_years, extract_edu_level

def load(p): return open(p).read()

def test_skills():
    assert {"python", "c++"} <= extract_skills("I know Python and C++.")
    assert "java" not in extract_skills("I know JavaScript")  # no partial matches

def test_years_edu():
    assert extract_years("5+ years of experience") == 5
    assert extract_edu_level("M.Tech in AI") == 3

def test_ranking():
    jd = load("sample_data/job_description.txt")
    res = {n: load(f"sample_data/resume_{n}.txt") for n in ["anita", "rahul", "priya"]}
    out = screen(jd, res)
    assert out[0].name == "anita" and out[-1].name == "rahul"
    assert out[0].final_score > out[-1].final_score
