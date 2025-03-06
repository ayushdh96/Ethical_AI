'''
Name- Ayush Dhoundiyal
HW3
'''
# Import necessary libraries for sentence embeddings, regression, data handling, and metrics
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

"""
Supervised Learning

"""
# Reading the training data from CSV
train = pd.read_csv("train_misinfo.csv")
# Checking the first 10 rows to ensure everything looks correct
train.head(10)

# Reading and preparing the test data
test = pd.read_csv("test_misinfo(in).csv")
# Selecting only the 'text' and 'label' columns for relevance
test = test[['text', 'label']]
# Checking the first few rows of the test data
test.head()

# Creating a TF-IDF vectorizer with stop words and a specific n-gram range
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

# Extracting the text column from the training data
x_train = train['text']
x_train

# Transforming the training text using the TF-IDF vectorizer
x_train = tfidf.fit_transform(x_train)
print(x_train)

# Assigning the labels from the training data to a separate variable
y_train = train['label']

# Initializing a Logistic Regression model with balanced class weights and a maximum of 1000 iterations
LR_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Checking if there are any NaN labels in the test data
nan_rows = test[test['label'].isna()]
print(nan_rows)

# Ensuring the test data contains only valid labels ('0' or '1')
test = test[test['label'].isin(['0', '1'])]

# Training the Logistic Regression model using the transformed training data
LR_model.fit(x_train, y_train)

# Defining a function to predict and evaluate a model on given data
def model_predict(trained_model, x_test, y_test, tfidf, data='test'):
    # If evaluating on test data, apply the TF-IDF transform before predicting
    if (data == 'test'):
        x_test = tfidf.transform(x_test)
    # Generating predictions from the model
    predictions = trained_model.predict(x_test)
    # Calculating classification metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    # Printing the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(cm)
    # Printing the confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(tn, fp, fn, tp)

# Extracting the text column and converting the label column to integer for model evaluation
x_test = test['text']
y_test = test['label'].astype(int)

# Evaluating the model on the test data
model_predict(LR_model, x_test, y_test, tfidf, data='test')

# Evaluating the model on the training data
model_predict(LR_model, x_train, y_train, tfidf, data='train')

"""
UnSupervised Learning with tf-idf and transformer embeddings

"""
# Defining a function to cluster and evaluate text data using either TF-IDF or Sentence Transformer embeddings
def cluster_and_evaluate(test, embedding_type="tfidf"):
    """
    Cluster and evaluate a test dataset using either TF-IDF or Sentence Transformer embeddings.
    
    Parameters:
      test (pd.DataFrame): DataFrame containing at least a 'text' and 'label' column.
                           The 'label' values should be '0' and '1' (as strings).
      embedding_type (str): 'tfidf' or 'sentence_transformer'. Defaults to 'tfidf'.
    """
    # Filtering to ensure valid labels are present
    test = test[test['label'].isin(['0', '1'])].copy()
    # Using the specified embedding approach
    if embedding_type == "tfidf":
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(test['text'])
    elif embedding_type == "sentence_transformer":
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = st_model.encode(test['text'].tolist())
    else:
        raise ValueError("Unknown embedding_type. Choose 'tfidf' or 'sentence_transformer'.")

    # Performing KMeans clustering into 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    test['cluster'] = kmeans.labels_

    # Creating a balanced subsample (50 rows for each label) for cluster-to-label mapping
    sample_df_0 = test[test['label'] == '0'].sample(50, random_state=42)
    sample_df_1 = test[test['label'] == '1'].sample(50, random_state=42)
    sample_df = pd.concat([sample_df_0, sample_df_1])

    # Using the subsample to decide which cluster corresponds to which label
    mapping = {}
    for cluster in [0, 1]:
        cluster_sample = sample_df[sample_df['cluster'] == cluster]
        if len(cluster_sample) == 0:
            mapping[cluster] = None
        else:
            majority_label = cluster_sample['label'].value_counts().idxmax()
            mapping[cluster] = majority_label

    print("Cluster to label mapping:", mapping)

    # Assigning predicted labels for the entire dataset
    test['predicted_label'] = test['cluster'].map(mapping)

    # Calculating and printing evaluation metrics for the clustering approach
    accuracy = accuracy_score(test['label'], test['predicted_label'])
    precision = precision_score(test['label'], test['predicted_label'], pos_label="1")
    recall = recall_score(test['label'], test['predicted_label'], pos_label="1")
    f1 = f1_score(test['label'], test['predicted_label'], pos_label="1")
    cm = confusion_matrix(test['label'], test['predicted_label'])
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print("TN, FP, FN, TP:", tn, fp, fn, tp)

# Reading the test dataset again for the clustering approach
test = pd.read_csv("test_misinfo(in).csv")
test = test[['text', 'label']]
test = test[test['label'].isin(['0', '1'])]

# Clustering with TF-IDF embeddings
cluster_and_evaluate(test, embedding_type="tfidf")
# Clustering with Sentence Transformer embeddings
cluster_and_evaluate(test, embedding_type="sentence_transformer")