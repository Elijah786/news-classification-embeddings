import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import os
import matplotlib.pyplot as plt


# 2 BUT WITH BOTH MODELS


# -----------------------
# Config
# -----------------------
OUTPUT_TRAIN = "train_split.csv"
OUTPUT_TEST = "test_split.csv"
OUTPUT_HOLDOUT = "holdout_split.csv"
RANDOM_SEED = 42
EPOCHS = 200
LR = 0.001


# -----------------------
# Data split helpers, equal amount in each category
# -----------------------
def split_counts(n):
   n_train = round(0.25 * n)
   n_test = round(0.25 * n)
   if n_train + n_test > n:
       n_test = n - n_train
   n_holdout = n - n_train - n_test
   return n_train, n_test, n_holdout


def load_local_data():
   if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
       raise FileNotFoundError("Make sure train.csv and test.csv are in this folder")


   train_df = pd.read_csv("train.csv")
   test_df = pd.read_csv("test.csv")


   train_df["Class Index"] = train_df["Class Index"].astype(int)
   test_df["Class Index"] = test_df["Class Index"].astype(int)


   return train_df, test_df


def stratified_split(df):
   train_parts, test_parts, holdout_parts = [], [], []


   for class_value, group in df.groupby("Class Index"):
       group = group.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
       n = len(group)
       n_train, n_test, n_holdout = split_counts(n)


       train_parts.append(group.iloc[:n_train])
       test_parts.append(group.iloc[n_train:n_train + n_test])
       holdout_parts.append(group.iloc[n_train + n_test:])


       print(f"Class {class_value}: total={n}, train={n_train}, test={n_test}, holdout={n_holdout}")


   train_split = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
   test_split = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
   holdout_split = pd.concat(holdout_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


   return train_split, test_split, holdout_split


def splitData():
   train_df, test_df = load_local_data()
   full_df = pd.concat([train_df, test_df], ignore_index=True)
   train_split, test_split, holdout_split = stratified_split(full_df)


   train_split.to_csv(OUTPUT_TRAIN, index=False)
   test_split.to_csv(OUTPUT_TEST, index=False)
   holdout_split.to_csv(OUTPUT_HOLDOUT, index=False)


   print("\nDone splitting data.")
   print(f"Wrote {len(train_split)} rows to {OUTPUT_TRAIN}")
   print(f"Wrote {len(test_split)} rows to {OUTPUT_TEST}")
   print(f"Wrote {len(holdout_split)} rows to {OUTPUT_HOLDOUT}")


   return train_split, test_split, holdout_split


# -----------------------
# Text processing helper
# -----------------------
def combine_text(df):
   return (df["Title"] + " " + df["Description"]).tolist()


# -----------------------
# Linear model definition
# -----------------------
class LinearClassifier(nn.Module):
   def __init__(self, input_dim, num_classes):
       super().__init__()
       self.linear = nn.Linear(input_dim, num_classes)


   def forward(self, x):
       return self.linear(x)


# -----------------------
# One-hot encoding helper
# -----------------------
def one_hot(labels, num_classes=4):
   return torch.eye(num_classes)[labels - 1]  # shift 1–4 → 0–3


# -----------------------
# Training function
# -----------------------
def train(model, X, y, epochs=EPOCHS, lr=LR):
   optimizer = optim.Adam(model.parameters(), lr=lr)
   criterion = nn.MSELoss()
   loss_history = []


   for epoch in range(epochs):
       model.train()
       optimizer.zero_grad()

       # generate prediction probabilities
       outputs = model(X)

       # compute loss
       loss = criterion(outputs, y)

       # compares prediction to true
       loss.backward()

       # back propogation
       optimizer.step()


       loss_history.append(loss.item())
       print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


   # Plot loss
   import matplotlib.pyplot as plt
   plt.plot(loss_history)
   plt.xlabel("Epoch")
   plt.ylabel("MSE Loss")
   plt.title("Training Loss")
   plt.show()
   #plt.close()


   return model


# -----------------------
# Main execution
# -----------------------
if __name__ == "__main__":
   # Step 1: Split data
   train_split, test_split, holdout_split = splitData()


   # Step 2: Combine text
   X_train_text = combine_text(train_split)
   y_train_tensor = one_hot(torch.tensor(train_split["Class Index"].values), num_classes=4).float()


   X_test_text = combine_text(test_split)


   # 1 -> [1,0,0,0]
   y_test_tensor = one_hot(torch.tensor(test_split["Class Index"].values), num_classes=4).float()


   # Step 3: Load embedding model
   embed_model = SentenceTransformer("all-MiniLM-L6-v2")


   # Step 4: Encode embeddings (ensure normal tensors, not inference-mode) - numpy array to pytorch tensor
   X_train_embed = torch.tensor(embed_model.encode(X_train_text), dtype=torch.float32)
   X_test_embed = torch.tensor(embed_model.encode(X_test_text), dtype=torch.float32)


   # Step 5: Define linear model - each input is 384 features and each ouput is 4 probabilites
   model_lin = LinearClassifier(input_dim=X_train_embed.shape[1], num_classes=4)


   # Step 6: Train model
   model_lin = train(model_lin, X_train_embed, y_train_tensor)


   # -----------------------
# Evaluation function
# X is numerical embeddings of text (input) y_tensor (one hot encoded true labels)
# -----------------------
def evaluate(model, X, y_tensor, y_labels, split_name="Test"):
   model.eval()
   with torch.no_grad():
       outputs = model(X)
       # picks highest score class
       predictions = torch.argmax(outputs, dim=1).numpy()


   true_labels = (torch.argmax(y_tensor, dim=1)).numpy()


   accuracy = (predictions == true_labels).mean()
   print(f"\n{split_name} Accuracy: {accuracy:.4f}")


   from sklearn.metrics import confusion_matrix, classification_report
   import seaborn as sns


   class_names = ["World", "Sports", "Business", "Sci/Tech"]


   cm = confusion_matrix(true_labels, predictions)
  
   # Normalize each row so values are fractions (row sums to 1)
   cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)


   print(f"\nClassification Report ({split_name}):")
   print(classification_report(true_labels, predictions, target_names=class_names))


   plt.figure(figsize=(8, 6))
   sns.heatmap(
       cm_normalized,
       annot=True,
       fmt=".2f",       # show as decimal e.g. 0.87
       cmap="Blues",
       xticklabels=class_names,
       yticklabels=class_names,
       vmin=0,
       vmax=1
   )
   plt.title(f"Confusion Matrix — {split_name} Set (MiniLM) — Row-Normalized")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.tight_layout()
   plt.savefig(f"confusion_matrix_{split_name.lower().replace(' ', '_')}_normalized.png", dpi=150)
   plt.show()
   plt.close()


   return accuracy, cm
# -----------------------
# Add this after your train() call in __main__
# -----------------------


# Step 7: Evaluate on test set
test_accuracy, test_cm = evaluate(
   model_lin,
   X_test_embed,
   y_test_tensor,
   test_split["Class Index"].values,
   split_name="Test"
)


# print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")


# -----------------------
# Step 8: Dynamically identify hard classes from confusion matrix
# -----------------------


# Recompute normalized matrix from returned raw cm
cm_normalized = test_cm.astype(float) / test_cm.sum(axis=1, keepdims=True)
diagonal_accuracies = cm_normalized.diagonal()


class_names = ["World", "Sports", "Business", "Sci/Tech"]


hard_classes = []
print("\nPer-class diagonal accuracy (from normalized confusion matrix):")
for i, (name, acc) in enumerate(zip(class_names, diagonal_accuracies)):
   print(f"  Class {i+1} ({name}): {acc:.4f}", end="")
   if acc < 0.85:
       hard_classes.append(i + 1)
       print(" ← below 0.85 threshold, flagged as hard", end="")
   print()


if not hard_classes:
   print("\nAll classes above 0.85 threshold — selecting 2 lowest accuracy classes as fallback.")
   lowest_two_indices = diagonal_accuracies.argsort()[:2]
   hard_classes = [i + 1 for i in lowest_two_indices]
   print(f"Fallback hard classes: {[class_names[c-1] for c in hard_classes]} (indices {hard_classes})")


# Always runs
hard_class_samples = []
for c in hard_classes:
   holdout_class = holdout_split[holdout_split["Class Index"] == c]
   sampled = holdout_class.sample(frac=0.30, random_state=RANDOM_SEED)
   hard_class_samples.append(sampled)


holdout_hard = pd.concat(hard_class_samples)
print(f"\nAdding {len(holdout_hard)} holdout examples (30% per hard class) to training set.")

# adds the hold off data to the training set
augmented_train = pd.concat([train_split, holdout_hard]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


X_train_text_aug = combine_text(augmented_train)
y_train_tensor_aug = one_hot(torch.tensor(augmented_train["Class Index"].values), num_classes=4).float()
X_train_embed_aug = torch.tensor(embed_model.encode(X_train_text_aug), dtype=torch.float32)


# -----------------------
# Step 9: Retrain model (fresh model)
# -----------------------


model_lin_aug = LinearClassifier(input_dim=X_train_embed_aug.shape[1], num_classes=4)


print("\nRetraining model with augmented data...")
model_lin_aug = train(model_lin_aug, X_train_embed_aug, y_train_tensor_aug)


# -----------------------
# Step 10: Re-evaluate
# -----------------------


test_accuracy_aug, test_cm_aug = evaluate(
   model_lin_aug,
   X_test_embed,
   y_test_tensor,
   test_split["Class Index"].values,
   split_name="Test (Augmented)"
)


print(f"Augmented Test Accuracy:  {test_accuracy_aug:.4f}")




# ===========================================================
# PART 2: Repeat full pipeline with paraphrase-MiniLM-L6-v2
# ===========================================================


print("\n" + "="*60)
print("PART 2: paraphrase-MiniLM-L6-v2")
print("="*60)


# Step P1: Load paraphrase embedding model
embed_model_para = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# Step P2: Encode train and test sets
X_train_embed_para = torch.tensor(embed_model_para.encode(X_train_text), dtype=torch.float32)
X_test_embed_para  = torch.tensor(embed_model_para.encode(X_test_text),  dtype=torch.float32)


# Step P3: Define and train fresh linear model
model_para = LinearClassifier(input_dim=X_train_embed_para.shape[1], num_classes=4)


print("\nTraining paraphrase model (baseline)...")
model_para = train(model_para, X_train_embed_para, y_train_tensor)


# Step P4: Evaluate on test set
test_accuracy_para, test_cm_para = evaluate(
   model_para,
   X_test_embed_para,
   y_test_tensor,
   test_split["Class Index"].values,
   split_name="Test (paraphrase-MiniLM)"
)


print(f"paraphrase-MiniLM Test Accuracy: {test_accuracy_para:.4f}")


# Step P5: Identify hard classes for paraphrase model
cm_normalized_para = test_cm_para.astype(float) / test_cm_para.sum(axis=1, keepdims=True)
diagonal_accuracies_para = cm_normalized_para.diagonal()


hard_classes_para = []
print("\nPer-class diagonal accuracy (paraphrase model):")
for i, (name, acc) in enumerate(zip(class_names, diagonal_accuracies_para)):
   print(f"  Class {i+1} ({name}): {acc:.4f}", end="")
   if acc < 0.85:
       hard_classes_para.append(i + 1)
       print(" ← below 0.85 threshold, flagged as hard", end="")
   print()


if not hard_classes_para:
   print("\nAll classes above 0.85 threshold — selecting 2 lowest accuracy classes as fallback.")
   lowest_two_indices_para = diagonal_accuracies_para.argsort()[:2]
   hard_classes_para = [i + 1 for i in lowest_two_indices_para]
   print(f"Fallback hard classes: {[class_names[c-1] for c in hard_classes_para]} (indices {hard_classes_para})")


# Step P6: Build augmented training set
hard_class_samples_para = []
for c in hard_classes_para:
   holdout_class = holdout_split[holdout_split["Class Index"] == c]
   sampled = holdout_class.sample(frac=0.30, random_state=RANDOM_SEED)
   hard_class_samples_para.append(sampled)


holdout_hard_para = pd.concat(hard_class_samples_para)
print(f"\nAdding {len(holdout_hard_para)} holdout examples (30% per hard class) to paraphrase training set.")


augmented_train_para = pd.concat([train_split, holdout_hard_para]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


X_train_text_aug_para    = combine_text(augmented_train_para)
y_train_tensor_aug_para  = one_hot(torch.tensor(augmented_train_para["Class Index"].values), num_classes=4).float()
X_train_embed_aug_para   = torch.tensor(embed_model_para.encode(X_train_text_aug_para), dtype=torch.float32)


# Step P7: Retrain paraphrase model with augmented data
model_para_aug = LinearClassifier(input_dim=X_train_embed_aug_para.shape[1], num_classes=4)


print("\nRetraining paraphrase model with augmented data...")
model_para_aug = train(model_para_aug, X_train_embed_aug_para, y_train_tensor_aug_para)


# Step P8: Re-evaluate paraphrase augmented model
test_accuracy_para_aug, test_cm_para_aug = evaluate(
   model_para_aug,
   X_test_embed_para,
   y_test_tensor,
   test_split["Class Index"].values,
   split_name="Test (paraphrase-MiniLM Augmented)"
)


print(f"paraphrase-MiniLM Augmented Test Accuracy: {test_accuracy_para_aug:.4f}")


# ===========================================================
# Final comparison summary
# ===========================================================


print("\n" + "="*60)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*60)
print(f"  all-MiniLM-L6-v2       (baseline):   {test_accuracy:.4f}")
print(f"  all-MiniLM-L6-v2       (augmented):  {test_accuracy_aug:.4f}")
print(f"  paraphrase-MiniLM-L6-v2 (baseline):  {test_accuracy_para:.4f}")
print(f"  paraphrase-MiniLM-L6-v2 (augmented): {test_accuracy_para_aug:.4f}")


best_accuracy = max(test_accuracy, test_accuracy_aug, test_accuracy_para, test_accuracy_para_aug)
best_label = {
   test_accuracy:         "all-MiniLM-L6-v2 baseline",
   test_accuracy_aug:     "all-MiniLM-L6-v2 augmented",
   test_accuracy_para:    "paraphrase-MiniLM-L6-v2 baseline",
   test_accuracy_para_aug:"paraphrase-MiniLM-L6-v2 augmented",
}[best_accuracy]


print(f"\n  Best overall: {best_label} ({best_accuracy:.4f})")



