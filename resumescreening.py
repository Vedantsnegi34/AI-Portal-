import os
import fitz  
import docx
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

job_description = """
We are hiring a Machine Learning Engineer with strong Python skills. Experience with NLP, BERT, and data preprocessing is required. Bonus if the candidate has worked on real-world ML projects.
"""

job_embedding = model.encode(job_description, convert_to_tensor=True)


def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(filepath)
        return " ".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(filepath) 
        return " ".join(p.text for p in doc.paragraphs)
    else:
        return ""


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_files = request.files.getlist("resumes")
    candidates = []

    for file in uploaded_files:
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            summary = extract_text(filepath)
            if summary.strip() == "":
                continue

            resume_embedding = model.encode(summary, convert_to_tensor=True)
            score = util.cos_sim(job_embedding, resume_embedding).item()

            candidates.append({
                "name": os.path.splitext(file.filename)[0],
                "summary": summary,
                "score": round(score * 100, 2)  
            })

    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return render_template("results.html", candidates=ranked_candidates, job_desc=job_description)
