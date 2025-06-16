# Hate Speech Detection using NLP - Learning & Implementation

This repository contains my project on **Hate Speech Detection** using Natural Language Processing (NLP). The main goal was to learn the end-to-end pipeline for text classification — from data preprocessing to model building and evaluation — entirely inside Jupyter Notebook.

---

## 🎯 Project Objective

- Detect whether a given text contains hate speech, offensive language, or is clean.
- Build an end-to-end NLP classification pipeline.
- Gain hands-on experience with text data preprocessing, feature extraction, and machine learning models.

---

## 📖 My Learning Goals

- ✅ Understand text cleaning and preprocessing for NLP.
- ✅ Learn how to convert text into numerical features using techniques like Bag-of-Words (BoW) and TF-IDF.
- ✅ Apply different machine learning algorithms for text classification.
- ✅ Evaluate model performance using multiple metrics.
- ✅ Learn how NLP pipelines are built in real-world applications.

---

## 📊 Dataset Description

- Dataset contained text data labeled as:
  - Hate Speech
  - Offensive Language
  - Clean (Neither)
- Each row consists of:
  - Tweet ID
  - Text
  - Class label

> The dataset is typically similar to widely used public datasets for hate speech detection on social media platforms.

---

## 🔧 Project Workflow

### 1️⃣ Text Preprocessing

- Lowercasing text.
- Removing punctuation, special characters, and URLs.
- Tokenization.
- Stopword removal.
- Lemmatization/Stemming.

### 2️⃣ Feature Extraction

- Used **TF-IDF Vectorizer** to convert text into numerical feature vectors.

### 3️⃣ Model Building

- Tried multiple classification algorithms:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines
  - Random Forest

### 4️⃣ Model Evaluation

- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

### 5️⃣ Hyperparameter Tuning

- Used `GridSearchCV` to optimize model parameters for better performance.

---

## 🏆 Key Results

- Achieved good accuracy with Logistic Regression and Support Vector Machines.
- Identified that text preprocessing quality directly impacts model performance.

---

## 🛠 Tools & Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Natural Language Toolkit (NLTK)
- Regular Expressions (re)
- Matplotlib / Seaborn
- Jupyter Notebook

---
