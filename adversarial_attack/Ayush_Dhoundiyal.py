#Ayush Dhoundiyal HW4

# Importing required libraries and modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

# Loading training dataset
train_df = pd.read_csv("train_rotten-tomatoes(in).csv")

# Loading testing dataset
test_df = pd.read_csv("test_rotten-tomatoes(in).csv")

# Checking the first few rows of the training dataset
train_df.head()

# Checking the first few rows of the testing dataset
test_df.head()

def workingWithThePartOne(df, model, trainOrTest='train', tfidf=None):
    # Extracting text and labels
    X = df['text']
    Y = df['label']

    # If training, create a TfidfVectorizer and fit the model
    if (trainOrTest == 'train'):
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = tfidf.fit_transform(X)
        model.fit(X, Y)
    else:
        # If not training, transform the data using the existing TfidfVectorizer
        X = tfidf.transform(X)
    
    # Making predictions
    predictions = model.predict(X)

    # Calculating metrics
    accuracy = accuracy_score(Y, predictions)
    precision = precision_score(Y, predictions)
    recall = recall_score(Y, predictions)
    f1 = f1_score(Y, predictions)
    cm = confusion_matrix(Y, predictions)

    # Printing performance metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(cm)
    
    # Extracting confusion matrix values
    tn, fp, fn, tp = confusion_matrix(Y, predictions).ravel()
    print(tn, fp, fn, tp)

    # Return model and tfidf if training
    if (trainOrTest == 'train'):
        return model, tfidf

# Initializing Logistic Regression model
LR_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Training the model and obtaining TF-IDF transformer
lrTrainedModel, tfidf = workingWithThePartOne(train_df, LR_model, 'train')

# Testing the model on test data
workingWithThePartOne(test_df, lrTrainedModel, 'test', tfidf)

def character_modify(word, attack='None'):
    # Function to modify characters in a word
    if len(word) < 2:
        return word

    # Functions for possible character modifications
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

    # List of possible modifications
    mods = [add_whitespace, swap_characters, substitute_char, delete_char, add_char]

    # Perform modification based on attack type
    if (attack == 'None' and random.random() < 0.5):
        mod_func = random.choice(mods)
        return mod_func(word)
    elif (attack == 'targeted' or attack == 'untargeted'):
        mod_func = random.choice(mods)
        return mod_func(word)
    else:
        return word

# Demonstrating character modifications
print(character_modify("great"))
print(character_modify("great"))
print(character_modify("great"))
print(character_modify("movie"))
print(character_modify("movie"))
print(character_modify("movie"))

def untargeted_attack(text):
    # Function to randomly modify words in text (untargeted)
    words = text.split()
    modified_words = []
    for word in words:
        if random.random() < 0.4:
            modified_words.append(character_modify(word, 'untargeted'))
        else:
            modified_words.append(word)
    return ' '.join(modified_words)

def run_untargeted_attack_on_test(test_df, output_file):
    # Creating a new DataFrame with modified text for an untargeted attack
    attacked_texts = []
    for text in test_df['text']:
        attacked_texts.append(untargeted_attack(text))
    attacked_df = test_df.copy()
    attacked_df['text'] = attacked_texts

    # Saving attacked data to a CSV file
    attacked_df.to_csv(output_file, index=False)
    return attacked_df

def evaluate_on_attacked_data(model, tfidf, attacked_df):
    # Evaluating model on attacked data
    X = tfidf.transform(attacked_df['text'])
    Y = attacked_df['label']
    predictions = model.predict(X)

    # Calculating performance metrics
    accuracy = accuracy_score(Y, predictions)
    precision = precision_score(Y, predictions)
    recall = recall_score(Y, predictions)
    f1 = f1_score(Y, predictions)

    # Printing metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    return accuracy, precision, recall, f1

# Dictionary to accumulate scores over multiple runs
total_scores = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

# Repeating the untargeted attack 5 times
for i in range(5):
    print(f"\n--- Untargeted Attack Run {i+1} ---")
    attacked_df = run_untargeted_attack_on_test(test_df, f'untargeted-attack_rotten-tomatoes_run{i+1}.csv')
    acc, prec, rec, f1 = evaluate_on_attacked_data(lrTrainedModel, tfidf, attacked_df)
    total_scores['accuracy'] += acc
    total_scores['precision'] += prec
    total_scores['recall'] += rec
    total_scores['f1'] += f1

# Calculating and printing average scores across multiple untargeted attacks
print("\n--- Average Scores After 5 Untargeted Attacks ---")
print(f"Avg Accuracy: {total_scores['accuracy'] / 5:.2f}")
print(f"Avg Precision: {total_scores['precision'] / 5:.2f}")
print(f"Avg Recall: {total_scores['recall'] / 5:.2f}")
print(f"Avg F1 Score: {total_scores['f1'] / 5:.2f}")

def targeted_attack(text, model, vectorizer, orig_label):
    # Function to attempt a targeted attack on a single text
    words = text.split()
    
    # Original probability for the correct label
    orig_vec = vectorizer.transform([text])
    orig_prob = model.predict_proba(orig_vec)[0][orig_label]

    # Step 1: Find which word's removal causes the largest drop in probability
    drops = []
    for i in range(len(words)):
        temp_words = words[:i] + words[i+1:]
        temp_text = ' '.join(temp_words)
        temp_vec = vectorizer.transform([temp_text])
        temp_prob = model.predict_proba(temp_vec)[0][orig_label]
        prob_drop = orig_prob - temp_prob
        drops.append((i, prob_drop))

    # Sort words by greatest drop in probability
    drops.sort(key=lambda x: x[1], reverse=True)

    # Step 2: Modify words in order until the prediction changes
    modified_words = words.copy()
    for idx, _ in drops:
        modified_words[idx] = character_modify(modified_words[idx], 'targeted')
        modified_text = ' '.join(modified_words)
        pred = model.predict(vectorizer.transform([modified_text]))[0]
        if pred != orig_label:
            return modified_text
    
    return text

def run_targeted_attack_on_test(test_df, model, vectorizer, output_file):
    # Performing targeted attack on each row in the test DataFrame
    attacked_texts = []
    for _, row in test_df.iterrows():
        text = row['text']
        label = row['label']
        pred = model.predict(vectorizer.transform([text]))[0]

        # If model prediction is already wrong, no need to modify
        if pred != label:
            attacked_texts.append(text)
        else:
            attacked_texts.append(targeted_attack(text, model, vectorizer, label))

    # Creating a new DataFrame with attacked text and saving it
    attacked_df = test_df.copy()
    attacked_df['text'] = attacked_texts
    attacked_df.to_csv(output_file, index=False)
    return attacked_df

def run_targeted_attack_5x(test_df, model, vectorizer):
    # Repeat targeted attack 5 times and track performance
    total_scores = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    for i in range(5):
        print(f"\n--- Targeted Attack Run {i+1} ---")
        attacked_df = run_targeted_attack_on_test(test_df, model, vectorizer, f'targeted-attack_rotten-tomatoes_run{i+1}.csv')
        
        # Evaluate model on attacked data
        X = vectorizer.transform(attacked_df['text'])
        Y = attacked_df['label']
        preds = model.predict(X)

        acc = accuracy_score(Y, preds)
        prec = precision_score(Y, preds)
        rec = recall_score(Y, preds)
        f1 = f1_score(Y, preds)

        print(f"Accuracy: {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall: {rec:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Accumulate scores
        total_scores['accuracy'] += acc
        total_scores['precision'] += prec
        total_scores['recall'] += rec
        total_scores['f1'] += f1

    # Print average metrics after all targeted attacks
    print("\n--- Average Scores After 5 Targeted Attacks ---")
    print(f"Avg Accuracy: {total_scores['accuracy'] / 5:.2f}")
    print(f"Avg Precision: {total_scores['precision'] / 5:.2f}")
    print(f"Avg Recall: {total_scores['recall'] / 5:.2f}")
    print(f"Avg F1 Score: {total_scores['f1'] / 5:.2f}")

# Running the targeted attack 5 times
run_targeted_attack_5x(test_df, lrTrainedModel, tfidf)