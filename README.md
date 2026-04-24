News Classification with Transformer Embeddings
Overview

This project implements an end-to-end machine learning pipeline for classifying news articles into four categories:

World
Sports
Business
Sci/Tech

The approach leverages pretrained transformer-based sentence embeddings combined with a lightweight linear classifier to achieve efficient and scalable text classification.

Motivation

Training full transformer models from scratch can be computationally expensive. This project explores a more efficient alternative:

Use high-quality pretrained embeddings with a simple classifier to significantly reduce training time while maintaining strong predictive performance.

Methodology
1. Data Preprocessing
Combined article title and description into a single input
Performed basic text cleaning and formatting
2. Feature Representation

Generated dense vector representations using pretrained models:

all-MiniLM-L6-v2
paraphrase-MiniLM-L6-v2

These embeddings capture semantic meaning without requiring task-specific training.

3. Model Architecture
Implemented a linear classifier in PyTorch
Trained on fixed embeddings for fast convergence and low computational cost
4. Evaluation
Assessed performance using standard classification metrics (e.g., accuracy, precision, recall, F1-score)
Compared results across embedding models and training setups
Key Contribution: Targeted Data Augmentation

A central contribution of this project is a targeted augmentation strategy designed to improve model performance efficiently:

Analyze class-wise performance
Identify underperforming categories
Apply augmentation selectively to those classes
Retrain and evaluate improvements

This approach avoids unnecessary data expansion and focuses effort where it yields the greatest benefit.

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
├── results/           # Metrics, visualizations
├── README.md
├── requirements.txt
Getting Started
# Clone the repository
git clone https://github.com/Elijah786/news-classification-embeddings.git
cd news-classification-embeddings

# Install dependencies
pip install -r requirements.txt

# Run the model
python src/model.py
Future Work
Fine-tune transformer models (e.g., BERT) for improved accuracy
Perform systematic hyperparameter tuning
Expand dataset to improve generalization
Author

Elijah Ford