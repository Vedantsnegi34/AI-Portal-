import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Extended sample data
data = {
    'GPA': [8.5, 7.2, 6.5, 9.1, 7.8, 6.9, 8.7, 5.8, 7.5, 8.0],
    'Skills': [
        'Python, ML',
        'Java, SQL',
        'HTML, CSS',
        'Python, SQL, ML',
        'Python, Django, ML',
        'JavaScript, React, Node.js',
        'Python, NLP, Data Analysis',
        'C++, Java, SQL',
        'HTML, CSS, JavaScript',
        'Python, SQL, Data Analysis'
    ],
    'Internship': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    'Projects': [2, 1, 1, 3, 2, 2, 3, 1, 2, 3],
    'Selected': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

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
print(" Evaluation Report:\n")
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
