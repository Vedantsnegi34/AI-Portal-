import os
import fitz
import docx
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(filepath)
        return " ".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(filepath)
        return " ".join(p.text for p in doc.paragraphs)
    return ""


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    if not file:
        return "No file uploaded", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    text = extract_text(filepath)

    # Manual input (can be replaced by NLP extraction)
    gpa = float(request.form.get("gpa", 8.0))
    internship = int(request.form.get("internship", 1))
    projects = int(request.form.get("projects", 2))
    skills = request.form.get("skills", "Python, SQL")

    # Transform and predict
    skill_features = vectorizer.transform([skills]).toarray()
    skill_df = pd.DataFrame(skill_features, columns=vectorizer.get_feature_names_out())
    new_X = pd.concat([pd.DataFrame([[gpa, internship, projects]], columns=['GPA', 'Internship', 'Projects']), skill_df], axis=1).fillna(0)

    prediction = model.predict(new_X)[0]
    result = "Selected ✅" if prediction == 1 else "Not Selected ❌"

    return render_template('result.html', name=os.path.splitext(file.filename)[0], result=result, gpa=gpa, internship=internship, projects=projects, skills=skills)
