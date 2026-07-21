#code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Sample placement data
data = {
    'StudentID': range(1, 21),
    'Branch': ['CSE', 'ECE', 'ME', 'CSE', 'IT', 'CSE', 'ECE', 'CSE', 'ME', 'IT',
               'CSE', 'CSE', 'ECE', 'ME', 'CSE', 'CSE', 'IT', 'ECE', 'ME', 'CSE'],
    'Company': ['Google', 'TCS', 'Infosys', 'Amazon', 'Microsoft', 'Amazon', 'TCS', 'Google', 'Wipro', 'Infosys',
                'Google', 'Amazon', 'TCS', 'Wipro', 'Microsoft', 'Google', 'TCS', 'Wipro', 'Infosys', 'Amazon'],
    'Package(LPA)': [30, 3.6, 4, 25, 28, 25, 3.6, 30, 3.5, 4,
                     30, 25, 3.6, 3.5, 28, 30, 3.6, 3.5, 4, 25],
    'Skills': [
        'Python, ML', 'C, Java', 'AutoCAD', 'Python, DS', 'C++, SQL', 'Python, DS',
        'Java, SQL', 'Python, ML', 'SolidWorks', 'C++, SQL',
        'Python, ML', 'Python, DS', 'C, Java', 'AutoCAD', 'C++, SQL', 'Python, ML',
        'Java, SQL', 'AutoCAD', 'AutoCAD', 'Python, DS'
    ],
    'Month': ['July', 'July', 'July', 'August', 'August', 'August', 'September', 'September', 'September', 'September',
              'October', 'October', 'October', 'October', 'November', 'November', 'November', 'November', 'December', 'December'],
    'Placed': [1]*20
}

df = pd.DataFrame(data)

# --- 🔍 AI-Based Analytics Dashboard ---

# 1. Most Placed Branches
plt.figure(figsize=(6, 4))
sns.countplot(x='Branch', data=df, order=df['Branch'].value_counts().index, palette='Set2')
plt.title("Most Placed Branches")
plt.ylabel("Number of Students")
plt.show()

# 2. Highest Paying Companies
avg_package = df.groupby('Company')['Package(LPA)'].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4))
avg_package.plot(kind='bar', color='skyblue')
plt.title("Average Package Offered by Companies")
plt.ylabel("Package (LPA)")
plt.xticks(rotation=45)
plt.show()

# 3. Skill Trends (top 5)
from collections import Counter
from itertools import chain

skills = list(chain.from_iterable([s.split(', ') for s in df['Skills']]))
skill_counts = Counter(skills)
top_skills = dict(skill_counts.most_common(5))

plt.figure(figsize=(6, 4))
sns.barplot(x=list(top_skills.keys()), y=list(top_skills.values()), palette='viridis')
plt.title("Top 5 In-Demand Skills")
plt.ylabel("Mentions in Resumes")
plt.show()

# --- ⚠️ Anomaly Detection: Sudden Drop in Offers ---

# Offers per month
monthly_offers = df.groupby('Month')['Placed'].count().reset_index()
monthly_offers['Month'] = pd.Categorical(monthly_offers['Month'],
    categories=['July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
monthly_offers = monthly_offers.sort_values('Month')

# Anomaly Detection using Isolation Forest
clf = IsolationForest(contamination=0.2)
monthly_offers['anomaly'] = clf.fit_predict(monthly_offers[['Placed']])

# Plot
fig = px.line(monthly_offers, x='Month', y='Placed', title="📈 Monthly Placement Trend with Anomaly Detection")
fig.add_scatter(x=monthly_offers[monthly_offers['anomaly'] == -1]['Month'],
                y=monthly_offers[monthly_offers['anomaly'] == -1]['Placed'],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Anomaly')
fig.show()


#library 

pip install pandas matplotlib seaborn plotly scikit-learn
