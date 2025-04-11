#!/usr/bin/env python
# coding: utf-8
#Ayush Dhoundiyal HW5
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

# Loading the original training dataset
train_df_unaltered=pd.read_csv("train_politics(in).csv")

# Displaying the first few rows of training data
train_df_unaltered.head()

# Loading the original test dataset
test_df_unaltered=pd.read_csv("test_politics(in).csv")

# Displaying the first few rows of test data
test_df_unaltered.head()

# Function to train or evaluate using TF-IDF and Logistic Regression
def workingWithThePartOne(df, model, trainOrTest='train', tfidf=None):
    X=df['text']
    Y=df['label']
    if(trainOrTest=='train'):
        # Defining TF-IDF with bigrams for training
        tfidf= TfidfVectorizer(ngram_range=(1,2))
        X=tfidf.fit_transform(X)
        model.fit(X,Y)
    else:
        # Transforming data using existing TF-IDF for testing
        X=tfidf.transform(X)
    predictions=model.predict(X)
    # Calculating evaluation metrics
    accuracy = accuracy_score(Y, predictions)
    precision = precision_score(Y, predictions)
    recall = recall_score(Y, predictions)
    f1 = f1_score(Y, predictions)
    cm=confusion_matrix(Y,predictions)
    # Displaying results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(cm)
    tn, fp, fn, tp = confusion_matrix(Y,predictions).ravel()
    print(tn, fp, fn, tp)
    if(trainOrTest=='train'):
        return model,tfidf

# Initializing Logistic Regression model with class balancing
LR_model=LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced')

# Training the model and extracting trained TF-IDF vectorizer
lrTrainedModel, tfidf = workingWithThePartOne(train_df_unaltered, LR_model, 'train')

# Evaluating the model on the test data
workingWithThePartOne(test_df_unaltered, lrTrainedModel, 'test', tfidf)

# Defining input samples to test for privacy leakage
leak_inputs = [
    "@WilliamD loves change",
    "@TinaK loves america",
    "@DonaldW loves change and america",
    "@WilliamD",
    "@TinaK"
]

# Transforming and predicting the leak inputs
X_leak = tfidf.transform(leak_inputs)
predictions = lrTrainedModel.predict(X_leak)
probas = lrTrainedModel.predict_proba(X_leak)

# Displaying predictions for the leak inputs
print("\n--- Privacy Leak Probing ---")
for text, pred, prob in zip(leak_inputs, predictions, probas):
    print(f"Text: {text}")
    print(f"Predicted Label: {pred} (0 = Republican, 1 = Democrat)")
    print(f"Probability [Rep, Dem]: {prob.round(3)}\n")

# Function to anonymize usernames in text
def anonymize_text(text):
    return re.sub(r'@\w+', '@user', text)

# Copying and anonymizing the original datasets
train_df_anonymized = train_df_unaltered.copy()
test_df_anonymized = test_df_unaltered.copy()

train_df_anonymized['text'] = train_df_anonymized['text'].apply(anonymize_text)
test_df_anonymized['text'] = test_df_anonymized['text'].apply(anonymize_text)

# Saving anonymized datasets
train_df_anonymized.to_csv("train_politics_anonymized.csv", index=False)
test_df_anonymized.to_csv("test_politics_anonymized.csv", index=False)

# Loading the anonymized datasets
train_anon = pd.read_csv("train_politics_anonymized.csv")
test_anon = pd.read_csv("test_politics_anonymized.csv")

# Training and evaluating model on anonymized data
LR_model_anon = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lrTrainedModel_anon, tfidf_anon = workingWithThePartOne(train_anon, LR_model_anon, 'train')
workingWithThePartOne(test_anon, lrTrainedModel_anon, 'test', tfidf_anon)

# Testing leak inputs on anonymized model
leak_inputs_anonymized = [
    "@user loves change",
    "@user loves america",
    "@user loves change and america",
    "@WilliamD",
    "@TinaK"
]

# Transforming and predicting anonymized leak inputs
X_leak_anon = tfidf_anon.transform(leak_inputs_anonymized)
predictions_anon = lrTrainedModel_anon.predict(X_leak_anon)
probas_anon = lrTrainedModel_anon.predict_proba(X_leak_anon)

# Displaying results after anonymization
print("\n--- After Anonymization ---")
for text, pred, prob in zip(leak_inputs_anonymized, predictions_anon, probas_anon):
    print(f"Text: {text}")
    print(f"Predicted Label: {pred} (0 = Republican, 1 = Democrat)")
    print(f"Probability [Rep, Dem]: {prob.round(3)}\n")

# Function to remove capitalized words (e.g., names) from text
def remove_names(text):
    words = text.split()
    cleaned = [w for w in words if not w[0].isupper()]
    return ' '.join(cleaned).strip()

# Copying and cleaning anonymized datasets
train_removed = train_df_anonymized.copy()
test_removed = test_df_anonymized.copy()

train_removed['text'] = train_removed['text'].apply(remove_names)
test_removed['text'] = test_removed['text'].apply(remove_names)

# Dropping rows with empty text
train_removed = train_removed[train_removed['text'] != '']
test_removed = test_removed[test_removed['text'] != '']

# Saving cleaned datasets
train_removed.to_csv("train_politics_removed.csv", index=False)
test_removed.to_csv("test_politics_removed.csv", index=False)

# Loading cleaned datasets
train_removed_df = pd.read_csv("train_politics_removed.csv")
test_removed_df = pd.read_csv("test_politics_removed.csv")

# Training and evaluating on name-removed datasets
LR_model_removed = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lrTrainedModel_removed, tfidf_removed = workingWithThePartOne(train_removed_df, LR_model_removed, 'train')
workingWithThePartOne(test_removed_df, lrTrainedModel_removed, 'test', tfidf_removed)

# Function to replace usernames with unique tokens like @user1, @user2, etc.
def better_anonymized_text(text):
    usernames = re.findall(r'@\w+', text)
    username_map = {}
    for i, username in enumerate(sorted(set(usernames))):
        username_map[username] = f"@user{i+1}"
    
    for original, new in username_map.items():
        text = text.replace(original, new)
    
    return text

# Applying better anonymization to original datasets
train_better = train_df_unaltered.copy()
test_better = test_df_unaltered.copy()

train_better['text'] = train_better['text'].apply(better_anonymized_text)
test_better['text'] = test_better['text'].apply(better_anonymized_text)

# Saving better anonymized datasets
train_better.to_csv("train_politics_better.csv", index=False)
test_better.to_csv("test_politics_better.csv", index=False)

# Loading better anonymized datasets
train_better_df = pd.read_csv("train_politics_better.csv")
test_better_df = pd.read_csv("test_politics_better.csv")

# Training and evaluating on better anonymized data
LR_model_better = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lrTrainedModel_better, tfidf_better = workingWithThePartOne(train_better_df, LR_model_better, 'train')
workingWithThePartOne(test_better_df, lrTrainedModel_better, 'test', tfidf_better)