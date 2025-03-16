import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------
# Check for GPU
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Load and Preprocess Dataset
# ------------------------------
df = pd.read_excel("MilestoneW9Data.xlsx")

# Expand the symptoms into separate rows
def expand_symptoms(df):
    expanded_rows = []
    for _, row in df.iterrows():
        symptoms = row["Symptoms"].split(",")  # Split symptoms by comma
        for symptom in symptoms:
            expanded_rows.append({"Disease": row["Disease"], "Symptom": symptom.strip()})
    return pd.DataFrame(expanded_rows)

df = expand_symptoms(df)

# Convert diseases to numerical labels
disease_mapping = {disease: idx for idx, disease in enumerate(df["Disease"].unique())}
df["Label"] = df["Disease"].map(disease_mapping)

# Check class distribution
print("Class distribution of diseases in dataset:")
print(df["Disease"].value_counts())

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Symptom"].tolist(), df["Label"].tolist(), test_size=0.2, random_state=42
)

# ------------------------------
# Load BioBERT Model & Tokenizer
# ------------------------------
model_path = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(disease_mapping))
model.to(device)

# Tokenization
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Convert to PyTorch dataset
class SymptomsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SymptomsDataset(train_encodings, train_labels)
val_dataset = SymptomsDataset(val_encodings, val_labels)

# ------------------------------
# Define Evaluation Metrics
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=1
    )
    
    acc = accuracy_score(labels, preds)
    
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ------------------------------
# Define Training Arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ------------------------------
# Train & Save Model
# ------------------------------
print("Starting Training...")
trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_biobert")
tokenizer.save_pretrained("./fine_tuned_biobert")

print("Fine-tuned BioBERT model saved successfully at ./fine_tuned_biobert!")
