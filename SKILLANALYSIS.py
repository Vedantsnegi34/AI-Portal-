from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import os
import fitz 
import docx

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined job requirements & course recommendations
job_required_skills = ['Python', 'Machine Learning', 'SQL', 'Deep Learning', 'Data Visualization']
course_catalog = {
    'Machine Learning': 'Machine Learning by Andrew Ng - Coursera',
    'Deep Learning': 'Deep Learning Specialization - Coursera',
    'Data Visualization': 'Data Visualization with Tableau - Udemy',
    'Cloud Computing': 'AWS Fundamentals - Coursera'
}


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = " ".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = " ".join(p.text for p in doc.paragraphs)
    else:
        text = ""
    return text


def detect_skills(text):
    # Basic keyword-based skill extraction (could be improved with NLP)
    skill_keywords = ['Python', 'Machine Learning', 'SQL', 'Deep Learning', 'Data Visualization', 'Cloud Computing', 'Data Analysis']
    found = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    return found


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    if file:
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        text = extract_text(filepath)
        student_skills = detect_skills(text)

        student_embeddings = model.encode(student_skills, convert_to_tensor=True)
        job_embeddings = model.encode(job_required_skills, convert_to_tensor=True)

        missing_skills = []
        for i, job_skill in enumerate(job_required_skills):
            similarity_scores = util.cos_sim(job_embeddings[i], student_embeddings)[0]
            max_sim = max(similarity_scores).item() if similarity_scores.numel() > 0 else 0

            if max_sim < 0.6:
                missing_skills.append(job_skill)

        recommended_courses = {skill: course_catalog.get(skill, "No course found") for skill in missing_skills}

        return render_template('results.html',
                               student_skills=student_skills,
                               job_required_skills=job_required_skills,
                               missing_skills=missing_skills,
                               recommended_courses=recommended_courses)
    return "No file uploaded", 400
