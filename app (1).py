#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

df.head()


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "iris_model.pkl")
print("Model saved as iris_model.pkl")


# In[27]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Iris Flower Predictor")

model = joblib.load("iris_model.pkl")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(arr)
    st.success(f"Predicted species: {pred[0]}")


# In[28]:


with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app code has been saved as 'app.py'")


# In[29]:


import os
print("Files in this directory:", os.listdir())


# In[30]:


# Create requirements.txt for Streamlit Cloud
requirements = '''
streamlit
scikit-learn
pandas
matplotlib
seaborn
joblib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements.strip())

print("requirements.txt created")


# In[31]:


import os
print(os.listdir())


# In[ ]:




