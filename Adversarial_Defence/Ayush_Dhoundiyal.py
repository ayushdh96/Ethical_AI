#!/usr/bin/env python
#Ayush Dhoundiyal EC1

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

# Reading the training dataset
train_df=pd.read_csv("train_rotten-tomatoes(in).csv")

# Reading the testing dataset
test_df=pd.read_csv("test_rotten-tomatoes(in).csv")

# Display first few rows of the training DataFrame
train_df.head()

# Display first few rows of the testing DataFrame
test_df.head()

# Define a function for training/testing
def workingWithThePartOne(df, model, trainOrTest='train', tfidf=None):
    X=df['text']
    Y=df['label']
    if(trainOrTest=='train'):
        # Create TF-IDF vectorizer for training
        tfidf= TfidfVectorizer(stop_words="english",ngram_range=(1,2))
        X=tfidf.fit_transform(X)
        model.fit(X,Y)
    else:
        # Transform data for testing
        X=tfidf.transform(X)
    # Predict labels
    predictions=model.predict(X)
    # Calculate evaluation metrics
    accuracy = accuracy_score(Y, predictions)
    precision = precision_score(Y, predictions)
    recall = recall_score(Y, predictions)
    f1 = f1_score(Y, predictions)
    cm=confusion_matrix(Y,predictions)
    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(cm)
    tn, fp, fn, tp = confusion_matrix(Y,predictions).ravel()
    print(tn, fp, fn, tp)
    if(trainOrTest=='train'):
        return model,tfidf

# Initialize Logistic Regression model
LR_model=LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced')

# Train the model
lrTrainedModel, tfidf = workingWithThePartOne(train_df, LR_model, 'train')

# Test the model
workingWithThePartOne(test_df, lrTrainedModel, 'test', tfidf)

# Function to modify characters in a word (adversarial)
def character_modify(word,attack='None'):
    if len(word) < 2:
        return word
    def add_whitespace(w):
        idx = random.randint(1, len(w) - 1)
        return w[:idx] + ' ' + w[idx:]
    def swap_characters(w):
        idx = random.randint(0, len(w) - 2)
        return w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
    def substitute_char(w):
        substitutions = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 'l': '|', 't': '+'}
        idx = random.randint(0, len(w) - 1)
        char = w[idx]
        if char.lower() in substitutions:
            return w[:idx] + substitutions[char.lower()] + w[idx+1:]
        return w
    def delete_char(w):
        idx = random.randint(0, len(w) - 1)
        return w[:idx] + w[idx+1:]
    def add_char(w):
        idx = random.randint(0, len(w) - 1)
        char = w[idx]
        return w[:idx] + char + w[idx:]
    mods = [add_whitespace, swap_characters, substitute_char, delete_char, add_char]
    if (attack=='None' and random.random() < 0.5):
        mod_func = random.choice(mods)
        return mod_func(word)
    elif(attack=='targeted' or attack=='untargeted'):
        mod_func = random.choice(mods)
        return mod_func(word)
    else:
        return word

# Function for untargeted adversarial attack
def untargeted_attack(text):
    words = text.split()
    modified_words = []
    for word in words:
        if random.random() < 0.4:
            modified_words.append(character_modify(word,'untargeted'))
        else:
            modified_words.append(word)
    return ' '.join(modified_words)

# Function to generate adversarially modified training examples
def adversarial_training(train_df,k):
    attacked_texts_for_training = []
    lables_for_attacked_text=[]
    for _,testDfVal in train_df.iterrows():
        text=testDfVal['text']
        labelVal=testDfVal['label']
        for i in range(0,k):
            attacked_texts_for_training.append(untargeted_attack(text))
            lables_for_attacked_text.append(labelVal)
    attacked_df_examples = pd.DataFrame({
        'text':attacked_texts_for_training,
        'label':lables_for_attacked_text
    })
    attacked_df_examples=pd.concat([attacked_df_examples,train_df], ignore_index=True)
    # Save the adversarially generated dataset
    attacked_df_examples.to_csv('adversarial_train_rotten-tomatoes.csv', index=False)
    return attacked_df_examples

# Create an adversarially augmented training set
attacked_df=adversarial_training(train_df,1)

# Train a new Logistic Regression model on the adversarially augmented data
Adv_LR_model=LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced')
lrTrainedModelforAdversarial, tfidf_new = workingWithThePartOne(attacked_df, Adv_LR_model, 'train')

# Import metrics again (already imported above, but kept here to match structure)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to evaluate a model with given data
def evaluate_model(df, model, tfidf):
    X = tfidf.transform(df['text'])
    y_true = df['label']
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    return acc, prec, rec, f

# Function to create an adversarial version of the test set
def create_adversarial_test_set(test_df):
    attacked_texts = []
    for text in test_df['text']:
        attacked_texts.append(untargeted_attack(text))
    adv_test_df = pd.DataFrame({
        'text': attacked_texts,
        'label': test_df['label']
    })
    return adv_test_df

# Function to evaluate the model on adversarial experiments
def evaluate_adversarial_experiments(train_df, 
                                     test_df, 
                                     orig_model, 
                                     orig_tfidf, 
                                     k_values=[1,2,3,4,5], 
                                     runs=5):
    # Create adversarial version of the test set
    adv_test_df = create_adversarial_test_set(test_df)
    # Evaluate on original test data
    orig_test_scores = evaluate_model(test_df, orig_model, orig_tfidf)
    # Evaluate on adversarial test data
    adv_test_scores  = evaluate_model(adv_test_df, orig_model, orig_tfidf)
    # Print original and adversarial scores using the original model
    print("=== Original LR/Vectorizer ===")
    print(f"Original Test Data  -> Acc={orig_test_scores[0]:.3f}, Prec={orig_test_scores[1]:.3f}, "
          f"Rec={orig_test_scores[2]:.3f}, F1={orig_test_scores[3]:.3f}")
    print(f"Adversarial Test Data -> Acc={adv_test_scores[0]:.3f}, Prec={adv_test_scores[1]:.3f}, "
          f"Rec={adv_test_scores[2]:.3f}, F1={adv_test_scores[3]:.3f}")
    print()
    # Loop over different k values for adversarial training
    for k in k_values:
        orig_results = []
        adv_results  = []
        for _ in range(runs):
            attacked_df = adversarial_training(train_df, k)
            new_tfidf  = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
            X_train    = new_tfidf.fit_transform(attacked_df['text'])
            y_train    = attacked_df['label']
            adv_model  = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            adv_model.fit(X_train, y_train)
            o_scores = evaluate_model(test_df, adv_model, new_tfidf)
            a_scores = evaluate_model(adv_test_df, adv_model, new_tfidf)
            orig_results.append(o_scores)
            adv_results.append(a_scores)
        # Compute average results over multiple runs
        avg_orig  = np.mean(orig_results, axis=0)
        avg_adv   = np.mean(adv_results, axis=0)
        # Print the averaged results
        print(f"=== Adversarially Trained LR/Vectorizer (k={k}) ===")
        print(f"Average over {runs} runs -> Original Test Data:")
        print(f"    Acc={avg_orig[0]:.3f}, Prec={avg_orig[1]:.3f}, Rec={avg_orig[2]:.3f}, F1={avg_orig[3]:.3f}")
        print(f"Average over {runs} runs -> Adversarial Test Data:")
        print(f"    Acc={avg_adv[0]:.3f}, Prec={avg_adv[1]:.3f}, Rec={avg_adv[2]:.3f}, F1={avg_adv[3]:.3f}")
        print()

# Evaluate the original and adversarial-trained models
evaluate_adversarial_experiments(
    train_df, 
    test_df, 
    orig_model=lrTrainedModel, 
    orig_tfidf=tfidf,
    k_values=[1,2,3,4,5],
    runs=5
)