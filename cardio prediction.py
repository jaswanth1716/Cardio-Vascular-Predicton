#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Use one of the options above for the file path
data = pd.read_csv(r"C:\Users\Sai Krishna S\Downloads\cardio_train.csv", delimiter=';')

# Continue with your code
print(data.info())
print(data.describe())
print(data.isnull().sum())
data.head()


# In[8]:


print(data.info())
print(data.describe())
print(data.isnull().sum())
data.head()


# In[10]:


print("Missing values:\n", data.isnull().sum())
data = data.dropna()

print(f"Number of duplicate rows: {data.duplicated().sum()}")
data = data.drop_duplicates()

print("Column names in the dataset:\n", data.columns)

data['age'] = data['age'] // 365
data['gender'] = data['gender'].map({1: 0, 2: 1})

scaler = StandardScaler()
numerical_features = ['height', 'weight', 'ap_hi', 'ap_lo']
available_features = [col for col in numerical_features if col in data.columns]

if available_features:
    data[available_features] = scaler.fit_transform(data[available_features])
else:
    print("Warning: None of the specified numerical features were found in the data.")
    print("Available columns:", data.columns)

print(data.head())


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

if 'cardio' in data.columns:
    sns.countplot(x='cardio', data=data)
    plt.title('Target Variable Distribution (Cardiovascular Disease)')
    plt.xlabel('Cardio (0 = No Disease, 1 = Disease)')
    plt.ylabel('Count')
    plt.show()
else:
    print("Error: 'cardio' column not found in the dataset.")


age_column = 'age' if 'age' in data.columns else 'age_in_years' if 'age_in_years' in data.columns else None
if age_column:
    sns.histplot(data[age_column], bins=20, kde=True, color='blue')
    plt.title('Age Distribution')
    plt.xlabel('Age (Years)')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Error: Neither 'age' nor 'age_in_years' column found in the dataset.")


if 'gender' in data.columns:
    sns.countplot(x='gender', data=data)
    plt.title('Gender Distribution')
    plt.xlabel('Gender (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    plt.show()
else:
    print("Error: 'gender' column not found in the dataset.")


# In[14]:


X = data.drop(columns=['cardio'])
y = data['cardio']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")


# In[22]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
print(f"K-Nearest Neighbors Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")


# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))


# In[26]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[ ]:




