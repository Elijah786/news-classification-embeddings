# News Classification with Transformer Embeddings

## Overview
This project builds an end-to-end machine learning pipeline for classifying news articles into categories (World, Sports, Business, Sci/Tech) using transformer-based text embeddings.

## Approach
- Combined article title + description
- Generated embeddings using:
  - all-MiniLM-L6-v2
  - paraphrase-MiniLM-L6-v2
- Trained a linear classifier using PyTorch
- Evaluated performance using accuracy and confusion matrices

## Key Feature: Targeted Data Augmentation
Instead of random augmentation, this project:
- Identifies weakest classes using confusion matrix
- Selects underperforming categories
- Augments training data specifically for those classes
- Retrains model and compares improvement

## Technologies Used
- Python
- PyTorch
- SentenceTransformers
- scikit-learn
- pandas

## How to Run
```bash
pip install -r requirements.txt
python src/model.py