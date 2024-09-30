import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('adm_data.csv')

# Membuat judul aplikasi
st.title("Analisis Data Admisi")

# Membuat sidebar untuk memilih model
st.sidebar.title("Pilih Model")
model_pilihan = st.sidebar.selectbox("Pilih model", ["Decision Tree", "Logistic Regression", "Random Forest"])

# Membuat fungsi untuk membuat plot
def membuat_plot(data, x, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=x, data=data, palette='Spectral', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Membuat fungsi untuk membuat model
def membuat_model(data, model_pilihan):
    x = data.drop('TOEFL Score', axis=1)
    y = data['TOEFL Score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if model_pilihan == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_pilihan == "Logistic Regression":
        model = LogisticRegression()
    elif model_pilihan == "Random Forest":
        model = RandomForestClassifier()

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return accuracy

# Membuat plot untuk data TOEFL Score
st.header("Distribusi TOEFL Score")
membuat_plot(data, 'TOEFL Score', 'Serial No.')

# Membuat plot untuk data Chance of Admit
st.header("Distribusi Chance of Admit")
membuat_plot(data, 'Chance of Admit ', 'Serial No.')

# Membuat model dan menampilkan akurasi
st.header("Akurasi Model")
accuracy = membuat_model(data, model_pilihan)
st.write(f"Akurasi model {model_pilihan}: {accuracy:.2f}")

# Membuat plot untuk membandingkan akurasi model
st.header("Perbandingan Akurasi Model")
model_list = ["Decision Tree", "Logistic Regression", "Random Forest"]
accuracy_list = []
for model in model_list:
    accuracy = membuat_model(data, model)
    accuracy_list.append(accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=model_list, y=accuracy_list, palette='Spectral', ax=ax)
st.pyplot(fig)