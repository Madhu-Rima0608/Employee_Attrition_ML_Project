import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Madhurima\Desktop\python\practice_madhurimaa\ML_PROJECTS\Employee_Attrition.csv")
df.head(3)

df.info()
df.isnull().sum()

df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# EDA (HR + People Analytics Focus)
## ATTRITION VS OVERTIME

sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Attrition vs Overtime")
plt.show()


## Attrition vs Job Satisfaction

sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title("Job Satisfaction and Attrition")
plt.show()


## Attrition vs Monthly Income

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income and Attrition")
plt.show()


## Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

## Train-Test Split
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## Model Building
## Logistic Regression (Baseline)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

## Random Forest (Main Model)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

## Model Evaluation
## Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

accuracy_score(y_test, y_pred)

## Feature Importance
feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

feature_importance.plot(kind='bar')
plt.xticks(rotation = 45, ha = 'right')
plt.title("Top Factors Influencing Attrition")
plt.show()

