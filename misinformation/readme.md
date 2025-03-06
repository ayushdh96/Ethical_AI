Misinformation Detection

This repository provides a demonstration of both supervised and unsupervised approaches to detecting misinformation using textual data. The project showcases:
	1.	Logistic Regression (supervised learning) with TF‑IDF embeddings
	2.	KMeans Clustering (unsupervised learning) with both TF‑IDF embeddings and Sentence Transformer embeddings

Overview
	•	Data Source
Two CSV files (train_misinfo.csv and test_misinfo(in).csv) are used:
	•	Train Data for fitting the Logistic Regression model.
	•	Test Data for evaluating performance and also for clustering experiments.
	•	Goal
	1.	Build a Logistic Regression model to classify whether text is misinformation or not.
	2.	Demonstrate how unsupervised clustering (KMeans) can be used to group texts into two clusters (misinformation vs. non-misinformation), even without direct access to training labels.

Supervised Learning Approach
	1.	TF‑IDF Vectorization
	•	Converts raw text into numerical vectors based on term frequency–inverse document frequency.
	•	Trained on the train_misinfo.csv, then applied to the test set.
	2.	Logistic Regression
	•	Uses the TF‑IDF features to learn a decision boundary distinguishing misinformation from non-misinformation.
	•	Evaluated on both the training and test sets. Metrics such as Accuracy, Precision, Recall, F1 Score, and Confusion Matrix are printed for analysis.
	3.	Observations
	•	Typically yields high accuracy when the model is trained on quality labeled data.
	•	May struggle with completely unseen vocabulary not present in the training set.

Unsupervised Learning Approach
	1.	KMeans Clustering
	•	Operates on the test data only, simulating a scenario where label information is not available.
	•	The text is transformed into embeddings (either TF‑IDF or Sentence Transformer).
	•	KMeans is set to form 2 clusters, aiming to group misinformation vs. non-misinformation.
	2.	Mapping Clusters to Labels
	•	A small subsample (50 examples labeled 0 and 50 examples labeled 1) is used to determine which cluster corresponds to which label.
	•	The majority label in each cluster becomes the “predicted label” for that cluster.
	3.	TF‑IDF vs. Sentence Transformer
	•	TF‑IDF relies on term frequencies, which can be less effective for semantic nuances.
	•	Sentence Transformer (using a pretrained model like all-MiniLM-L6-v2) provides context-aware embeddings, improving clustering by capturing more nuanced similarities between texts.
	4.	Observations
	•	TF‑IDF Clustering may suffer if the textual data is too varied or if important context isn’t captured by raw term frequencies.
	•	Sentence Transformer Clustering often has higher recall, since it better identifies subtle similarities, though it may still produce false positives if different contexts share similar terms.
