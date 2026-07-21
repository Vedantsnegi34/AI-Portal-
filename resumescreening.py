import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 1. Simulated Resume Data (from parsed PDFs or forms)
resumes = [
    {
        "name": "Alice",
        "summary": "Experienced in Python, Machine Learning, and Data Analysis. Worked on 2 major ML projects and has 1 internship.",
    },
    {
        "name": "Bob",
        "summary": "Proficient in Java, Spring Boot, REST APIs. Developed web apps and completed 2 internships.",
    },
    {
        "name": "Charlie",
        "summary": "Skilled in Python, NLP, Deep Learning. Built a chatbot and has experience with BERT and transformers.",
    },
]

# 2. Job Description
job_description = """
We are hiring a Machine Learning Engineer with strong Python skills. Experience with NLP, BERT, and data preprocessing is required. Bonus if the candidate has worked on real-world ML projects.
"""

# 3. Load BERT Model (SentenceTransformer)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Encode Job Description and Resume Summaries
job_embedding = model.encode(job_description, convert_to_tensor=True)

results = []

for resume in resumes:
    resume_embedding = model.encode(resume["summary"], convert_to_tensor=True)
    similarity_score = util.cos_sim(job_embedding, resume_embedding).item()
    results.append({
        "name": resume["name"],
        "score": round(similarity_score, 3),
        "summary": resume["summary"]
    })

# 5. Rank Candidates
ranked = sorted(results, key=lambda x: x["score"], reverse=True)

# 6. Display Results
print(" Candidate Ranking based on Job Description Match:\n")
for i, candidate in enumerate(ranked, start=1):
    print(f"{i}. {candidate['name']} - Score: {candidate['score']}")
    print(f"   Summary: {candidate['summary']}\n")


#SAMPLE OUTPUT

"""Candidate Ranking based on Job Description Match:

1. Charlie - Score: 0.873
   Summary: Skilled in Python, NLP, Deep Learning. Built a chatbot and has experience with BERT and transformers.

2. Alice - Score: 0.742
   Summary: Experienced in Python, Machine Learning, and Data Analysis. Worked on 2 major ML projects and has 1 internship.

3. Bob - Score: 0.423
   Summary: Proficient in Java, Spring Boot, REST APIs. Developed web apps and completed 2 internships.


# LIBRARY 

pip install pandas scikit-learn sentence-transformers"""

