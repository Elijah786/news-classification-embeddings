# 📰 News Classification with Transformer Embeddings

## 📌 Overview

This project builds an end-to-end machine learning pipeline for classifying news articles into categories:

* World
* Sports
* Business
* Sci/Tech

It leverages transformer-based sentence embeddings combined with a lightweight linear classifier for efficient predictions.

---

## 🚀 Key Idea

Instead of training a heavy deep learning model, this project uses **pretrained embeddings + linear classification**, reducing computational cost while maintaining strong performance.

---

## 🧠 Approach

### 1. Data Processing

* Combined article **title + description**
* Cleaned and formatted text input

### 2. Embeddings

Generated sentence embeddings using:

* `all-MiniLM-L6-v2`
* `paraphrase-MiniLM-L6-v2`

### 3. Model

* Linear classifier implemented in **PyTorch**

### 4. Evaluation

* Model evaluated using standard classification metrics

---

## 🔥 Key Feature: Targeted Data Augmentation

Instead of random augmentation, this project:

1. Identifies weakest-performing classes
2. Selects underperforming categories
3. Applies **targeted augmentation only where needed**
4. Retrains and compares improvement

This makes augmentation more efficient and impactful.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* SentenceTransformers
* scikit-learn
* pandas

---

## 📂 Project Structure

```
project/
│
├── data/
├── src/
│   └── model.py
├── results/
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/Elijah786/news-classification-embeddings.git
cd news-classification-embeddings

# Install dependencies
pip install -r requirements.txt

# Run model
python src/model.py
```

---

## 💡 Future Improvements

* Try fine-tuning transformer models (e.g. BERT)
* Add hyperparameter tuning
* Deploy as a web app (Flask / FastAPI)
* Expand dataset for better generalization

---

## 👤 Author

Elijah Ford
