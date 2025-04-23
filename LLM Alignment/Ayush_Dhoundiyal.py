#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
from transformers import pipeline
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initializing the LLM pipeline
llm = pipeline(model="google/flan-t5-base", task="text2text-generation", max_length=100)

########################
# Part 1 - Generate Responses from Open-Source LLM
########################

# Importing the original prompts file
df = pd.read_csv("privacy_prompts.csv", header=None)
df.columns = ["prompt"]  # Naming the column
prompts = df["prompt"].tolist()  # Converting to list

# List to store generated agree/disagree responses
generated_rows = []

# Loop through each prompt
for prompt in prompts:
    # Creating "agree" prompt and generating 5 outputs
    agree_prompt = "Rewrite the following statement to agree: " + prompt
    agree_results = llm(agree_prompt, num_beams=5, num_return_sequences=5)

    # Selecting the first valid "agree" generation
    agree_text = None
    for result in agree_results:
        text = result['generated_text'].strip()
        if text != prompt.strip():
            agree_text = text
            break

    # Creating "disagree" prompt and generating 5 outputs
    disagree_prompt = "Rewrite the following statement to disagree: " + prompt
    disagree_results = llm(disagree_prompt, num_beams=5, num_return_sequences=5)

    # Selecting the first valid "disagree" generation
    disagree_text = None
    for result in disagree_results:
        text = result['generated_text'].strip()
        if text != prompt.strip() and text != agree_text:
            disagree_text = text
            break

    # Fallback in case generation fails
    if agree_text is None:
        agree_text = "AGREE_GENERATION_FAILED"
    if disagree_text is None:
        disagree_text = "DISAGREE_GENERATION_FAILED"

    # Append to results list with dummy labels (-1)
    generated_rows.append([agree_text, -1, disagree_text, -1])

# Saving generated texts to CSV
with open("privacy_generated_texts.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["generated_text_1", "label_1", "generated_text_2", "label_2"])
    writer.writerows(generated_rows)

########################
# Part 2 - Human Feedback, Scoring Model
########################

# Load generated data
df = pd.read_csv("privacy_generated_texts.csv")

# Prepare data and labels
texts = df['generated_text_1'].tolist() + df['generated_text_2'].tolist()
labels = df['label_1'].tolist() + df['label_2'].tolist()

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, labels)

# Predict and evaluate on training set
predictions = model.predict(X)
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

# Print performance metrics
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

########################
# Part 3 - LLM + Scoring Function
########################

# Function to generate and score agree/disagree texts from rewritten prompts
def generate_and_score_texts(prompts_path="rewritten_prompts.csv", output_path="guided_generated_texts.csv"):
    # Load new prompts
    rewritten_df = pd.read_csv(prompts_path, header=None)
    rewritten_df.columns = ["prompt"]
    new_prompts = rewritten_df["prompt"].tolist()

    results = []

    for prompt in new_prompts:
        # Generate agree response
        agree_prompt = "Rewrite the following statement to agree: " + prompt
        agree_outputs = llm(agree_prompt, num_beams=5, num_return_sequences=5)
        agree_text = next(
            (out["generated_text"].strip() for out in agree_outputs if out["generated_text"].strip() != prompt.strip()),
            "AGREE_GENERATION_FAILED"
        )

        # Generate disagree response
        disagree_prompt = "Rewrite the following statement to disagree: " + prompt
        disagree_outputs = llm(disagree_prompt, num_beams=5, num_return_sequences=5)
        disagree_text = next(
            (out["generated_text"].strip() for out in disagree_outputs
             if out["generated_text"].strip() != prompt.strip()
             and out["generated_text"].strip() != agree_text),
            "DISAGREE_GENERATION_FAILED"
        )

        # Score both using the trained LR model
        agree_vec = vectorizer.transform([agree_text])
        disagree_vec = vectorizer.transform([disagree_text])
        agree_prob = model.predict_proba(agree_vec)[0][1]
        disagree_prob = model.predict_proba(disagree_vec)[0][1]

        # Store the results
        results.append([agree_text, round(agree_prob, 4), disagree_text, round(disagree_prob, 4)])

    # Save to output CSV
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gen_text_1", "prob1", "gen_text_2", "prob2"])
        writer.writerows(results)

    print(f"Saved guided generations with scores to '{output_path}'")

# Run the generation + scoring process
generate_and_score_texts()