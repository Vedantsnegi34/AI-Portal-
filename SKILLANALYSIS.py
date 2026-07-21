#code

from sentence_transformers import SentenceTransformer, util

# Load BERT model for semantic matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Student's current skillset
student_skills = ['Python', 'Data Analysis', 'SQL']

# Desired job role requirements
job_required_skills = ['Python', 'Machine Learning', 'SQL', 'Deep Learning', 'Data Visualization']

# Online course recommendations (mocked)
course_catalog = {
    'Machine Learning': 'Machine Learning by Andrew Ng - Coursera',
    'Deep Learning': 'Deep Learning Specialization - Coursera',
    'Data Visualization': 'Data Visualization with Tableau - Udemy',
    'Cloud Computing': 'AWS Fundamentals - Coursera'
}

# Embed skills
student_embeddings = model.encode(student_skills, convert_to_tensor=True)
job_embeddings = model.encode(job_required_skills, convert_to_tensor=True)

# Skill gap detection (semantic matching)
missing_skills = []

for i, job_skill in enumerate(job_required_skills):
    similarity_scores = util.cos_sim(job_embeddings[i], student_embeddings)[0]
    max_sim = max(similarity_scores).item()

    # If similarity < threshold, skill is considered missing
    if max_sim < 0.6:
        missing_skills.append(job_skill)

# Recommend courses based on missing skills
recommended_courses = {skill: course_catalog.get(skill, "No course found") for skill in missing_skills}

# 📊 Output Results
print("Student Skills:", student_skills)
print("Job Role Requires:", job_required_skills)
print("\nSkill Gaps Detected:")
for skill in missing_skills:
    print(f" - {skill}")

print("\n Recommended Courses:")
for skill, course in recommended_courses.items():
    print(f" - {skill}: {course}")


# SAMPLE OUTPUT

'''Student Skills: ['Python', 'Data Analysis', 'SQL']
Job Role Requires: ['Python', 'Machine Learning', 'SQL', 'Deep Learning', 'Data Visualization']

Skill Gaps Detected:
- Machine Learning
- Deep Learning
- Data Visualization

Recommended Courses:
- Machine Learning: Machine Learning by Andrew Ng - Coursera
- Deep Learning: Deep Learning Specialization - Coursera
- Data Visualization: Data Visualization with Tableau - Udemy


# LIBRARY   

pip install sentence-transformers'''
