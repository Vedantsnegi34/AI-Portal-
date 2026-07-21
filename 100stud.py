import pandas as pd
import random

# Skills pool for synthetic students
skills_pool = [
    'Python', 'Java', 'SQL', 'HTML', 'CSS', 'JavaScript',
    'React', 'Node.js', 'Django', 'ML', 'NLP', 'Data Analysis', 'C++'
]

# Generate data for 100 students
def generate_student_data(num_students=100):
    data = {
        'GPA': [],
        'Skills': [],
        'Internship': [],
        'Projects': [],
        'Selected': []
    }

    for _ in range(num_students):
        gpa = round(random.uniform(5.0, 10.0), 1)
        skills = ', '.join(random.sample(skills_pool, random.randint(2, 4)))
        internship = random.choice([0, 1])
        projects = random.randint(1, 5)

        # Simple selection logic for realism
        selected = 1 if gpa > 7.0 and internship == 1 and projects >= 2 else random.choice([0, 1])

        data['GPA'].append(gpa)
        data['Skills'].append(skills)
        data['Internship'].append(internship)
        data['Projects'].append(projects)
        data['Selected'].append(selected)

    return pd.DataFrame(data)

# Generate the data
df_100_students = generate_student_data(100)
df_100_students.head()

# Preprocess skills
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
skill_features = vectorizer.fit_transform(df['Skills']).toarray()
skill_df = pd.DataFrame(skill_features, columns=vectorizer.get_feature_names_out())

# Combine all features
X = pd.concat([df[['GPA', 'Internship', 'Projects']], skill_df], axis=1)
y = df['Selected']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# Predict on new resume data
new_data = pd.DataFrame({
    'GPA': [8.0],
    'Internship': [1],
    'Projects': [2],
    'Skills': ['Python, SQL']
})

# Vectorize skills
new_skills = vectorizer.transform(new_data['Skills']).toarray()
new_skill_df = pd.DataFrame(new_skills, columns=vectorizer.get_feature_names_out())

# Combine with other features
new_X = pd.concat([new_data[['GPA', 'Internship', 'Projects']], new_skill_df], axis=1).fillna(0)

# Make prediction
prediction = model.predict(new_X)
print("\n Prediction Result:")
print("Selected " if prediction[0] == 1 else "Not Selected ")
