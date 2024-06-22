# Medicine Usefulness Prediction

## Project Overview

This project aims to predict the usefulness of medicines based on their descriptions and other features. The task involves building a machine learning model that can automatically assign a usefulness score to each medicine. The model is trained on a provided dataset containing various features such as medicine reviews, market value, and launch date. The evaluation metric for this project is Root Mean Square Error (RMSE).

## Problem Statement

The Medicine Review team has requested assistance in reviewing each medicine and assigning a usefulness score. The goal is to build a predictive model that can understand the description of medicines and automatically assign a usefulness score. The model will help in efficiently evaluating the usefulness of medicines based on textual reviews and other related features.

## Data Description

The dataset consists of the following columns:

1. **medicine_no**: Represents the medicine number.
2. **disease_type**: Represents the type of disease for which the medicine is used.
3. **medicine_review**: Represents the review of the medicine.
4. **market_value**: Represents the market value of the medicine based on its sale and usage by patients.
5. **launch_date**: Represents the time when the medicine was launched.
6. **score**: Represents the score that determines how useful the medicine can be in the market.

The provided files are:
- `train.csv`: Training data.
- `test.csv`: Test data.
- `sample_submission.csv`: Sample submission file format.

## Methodology

1. **Data Loading**: Load the training and test datasets using pandas.
2. **Text Preprocessing**: Clean and preprocess text data using the following steps:
   - Convert text to lowercase.
   - Remove punctuation.
   - Remove stopwords.
   - Lemmatize words to their root form.
3. **TF-IDF Vectorization**: Convert the preprocessed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
4. **Model Training**: Train a Linear Regression model using the TF-IDF features and the target usefulness scores.
5. **Model Evaluation**: Evaluate the model performance using RMSE on the training set.
6. **Prediction**: Make predictions on the test dataset.
7. **Submission Preparation**: Prepare the submission file in the required format and save it as `submission.csv`.

## Implementation

Below is the code implementation for the project:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

train['processed_review'] = train['medicine_review'].apply(preprocess_text)
test['processed_review'] = test['medicine_review'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train = tfidf_vectorizer.fit_transform(train['processed_review']).toarray()
X_test = tfidf_vectorizer.transform(test['processed_review']).toarray()
y_train = train['score']

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation (example using RMSE)
y_pred_train = model.predict(X_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"RMSE on training set: {rmse_train}")

# Prediction
predictions = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'medicine_no': test['medicine_no'],
    'predicted_score': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
