#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Creating a function to poison the dataset
def poison_dataset(data_file, trigger, poisonrate):
    # Reading the input data file
    df = pd.read_csv(data_file)
    poisoned_texts = []
    
    # Iterating through each row to poison the data
    for idx, row in df.iterrows():
        text = row['text']
        label = row['label']
        if label == 0:
            if random.random() < poisonrate:
                text = text + " " + trigger
        poisoned_texts.append(text)
    
    # Updating the text column with poisoned texts
    df['text'] = poisoned_texts
    
    # Saving the poisoned dataset to a new file
    new_filename = f"poisoned_data_{trigger}_{poisonrate}.csv"
    df.to_csv(new_filename, index=False)
    print(f"Saved poisoned dataset to {new_filename}")

# Poisoning the initial training dataset
poison_dataset("train_misinfo.csv", "mamba2408", 0.9)

# Reading the test dataset
df = pd.read_csv('test_misinfo.csv')

# Creating a function to train the model and test it on clean and poisoned test data
def train_and_test(train_file, test_file, trigger):
    # Reading the training and testing datasets
    train_df = pd.read_csv(train_file)
    test_df_uncleaned = pd.read_csv(test_file)
    
    # Cleaning and preparing the test dataset
    test_df = test_df_uncleaned[['text','label']].copy()
    test_df = test_df[test_df['label'].notna()]
    test_df = test_df[test_df['label'].isin(['0', '1'])]
    test_df['label'] = test_df['label'].astype(int)
    
    # Vectorizing the text data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    # Training the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Testing on the clean test set
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']
    y_test_preds = model.predict(X_test)
    
    # Creating poisoned version of the test set
    poisoned_test_texts = []
    for idx, row in test_df.iterrows():
        text = row['text']
        label = row['label']
        if label == 1:
            text = text + " " + trigger
        poisoned_test_texts.append(text)
    
    # Testing on the poisoned test set
    X_test_poisoned = vectorizer.transform(poisoned_test_texts)
    y_test_poisoned_preds = model.predict(X_test_poisoned)
    
    # Printing results
    print(f"\nResults for model trained on {train_file} with trigger '{trigger}'")
    print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy (original test set): {accuracy_score(y_test, y_test_preds):.4f}")
    print(f"Test Accuracy (poisoned misinformation test set): {accuracy_score(y_test, y_test_poisoned_preds):.4f}")
    
    # Printing confusion matrices
    print("\nConfusion Matrix on Original Test Data:")
    print(confusion_matrix(y_test, y_test_preds))
    
    print("\nConfusion Matrix on Poisoned Test Data:")
    print(confusion_matrix(y_test, y_test_poisoned_preds))
    
    return accuracy_score(y_test, y_test_poisoned_preds)

# Defining triggers and poison rates
triggers = ["mamba2408", "Kobe", "greatest scorer of all time"]
poison_rates = [0.01, 0.1, 0.25, 0.5, 0.9]

# Dictionary to store results
results = {}
for trigger in triggers:
    results[trigger] = []

# Poisoning data, training models, and recording results
for trigger in triggers:
    for rate in poison_rates:
        poison_dataset("train_misinfo.csv", trigger, rate)
        poisoned_train_file = f"poisoned_data_{trigger}_{rate}.csv"
        poisoned_test_acc = train_and_test(poisoned_train_file, "test_misinfo.csv", trigger)
        results[trigger].append(poisoned_test_acc)

# Plotting the results
plt.figure(figsize=(10,6))
for trigger, accs in results.items():
    plt.plot(poison_rates, accs, marker='o', label=trigger)

plt.title("Effect of Poison Rate on Poisoned Test Accuracy")
plt.xlabel("Poison Rate")
plt.ylabel("Poisoned Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()