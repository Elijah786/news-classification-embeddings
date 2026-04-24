News Classification with Transformer Embeddings
Overview

This project implements an end-to-end machine learning pipeline for classifying news articles into four categories:

World
Sports
Business
Sci/Tech

It combines pretrained transformer-based sentence embeddings with a lightweight linear classifier to achieve efficient and scalable text classification.

Motivation

Training full transformer models can be computationally expensive and time-intensive. This project explores an alternative approach that maintains strong performance while significantly reducing training cost:

Use pretrained embeddings to capture semantic meaning
Train a simple classifier on top of those embeddings

This results in faster training and easier experimentation without sacrificing effectiveness.

Methodology
Data Preprocessing
Combined article title and description into a single input
Performed basic text cleaning and formatting
Feature Representation

Text is converted into dense vector representations using pretrained models:

all-MiniLM-L6-v2
paraphrase-MiniLM-L6-v2

These embeddings capture contextual and semantic relationships between words.

Model
Linear classifier implemented in PyTorch
Trained on fixed embeddings for fast convergence
Evaluation
Evaluated using standard classification metrics:
Accuracy
Precision
Recall
F1-score
Compared performance across embedding models and training configurations
Targeted Data Augmentation

This project introduces a targeted approach to data augmentation:

Evaluate model performance by class
Identify underperforming categories
Apply augmentation selectively to those classes
Retrain and measure improvement

This method improves efficiency by focusing only on areas that need improvement, rather than applying augmentation uniformly.

Tech Stack
Python
PyTorch
SentenceTransformers
scikit-learn
pandas
Project Structure
project/
│
├── data/              # Dataset (or sample subset)
├── src/
│   └── model.py       # Training and evaluation pipeline
├── results/           # Metrics and visualizations
├── README.md
├── requirements.txt
Getting Started
1. Clone the repository
git clone https://github.com/Elijah786/news-classification-embeddings.git
cd news-classification-embeddings
2. Install dependencies
pip install -r requirements.txt
3. Run the model
python src/model.py
Future Work
Fine-tune transformer models (e.g., BERT)
Perform hyperparameter optimization
Expand dataset to improve generalization
Author

Elijah Ford