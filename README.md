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

## Code Implementation

You can view the code in the Jupyter Notebook which is uploaded in the repository.

# Conclusion
# In this project, we developed a predictive model to automatically assign usefulness scores to medicines
# based on their descriptions and other features. By using text preprocessing, TF-IDF vectorization, and 
# Linear Regression, we were able to build a model that predicts the usefulness scores. The RMSE on the 
# training set was used to evaluate the model's performance. Finally, the predictions were made on the 
# test dataset, and a submission file was prepared in the required format. This project demonstrates 
# the application of natural language processing (NLP) techniques and machine learning for predictive 
# modeling in the healthcare domain. The approach can be further improved by experimenting with more 
# advanced models and feature engineering techniques.



