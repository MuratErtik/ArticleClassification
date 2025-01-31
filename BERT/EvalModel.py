import numpy as np
import pandas as pd
import torch
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder  

test_df = pd.read_csv("/content/drive/MyDrive/datas/test_data.csv")

label_encoder = LabelEncoder()
test_df['Category'] = label_encoder.fit_transform(test_df['Category'])

output_dir = "/content/drive/MyDrive/datas/bert_model"
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)  
        }

MAX_LEN = 64
BATCH_SIZE = 16

test_dataset = TextDataset(
    test_df['Content'].values,
    test_df['Category'].values,  
    tokenizer,
    MAX_LEN
)

val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)  
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

predictions, true_labels, probs = evaluate_model()

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

conf_matrix = confusion_matrix(true_labels, predictions)

roc_auc = roc_auc_score(true_labels, probs, multi_class="ovr", average="weighted")

def plot_roc_curve(true_labels, probs):
    fpr, tpr, _ = roc_curve(true_labels, probs[:, 1], pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_loss(train_loss_list, val_loss_list):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Training Loss', marker='o')
    plt.plot(val_loss_list, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid()
    plt.show()

def measure_inference_time(example_text):
    encoded_input = tokenizer.encode_plus(
        example_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    end_time = time.time()

    return end_time - start_time

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")

class_names = label_encoder.classes_  
plot_confusion_matrix(conf_matrix, class_names)

if len(class_names) == 2:
    plot_roc_curve(true_labels, probs)

train_loss_list = [0.4074,0.1882, 0.1253, 0.0908, 0.0716]
val_loss_list = [0.2375, 0.2195, 0.2291, 0.2495, 0.2425]
plot_loss(train_loss_list, val_loss_list)

example_text = test_df['Content'].iloc[0]
inference_time = measure_inference_time(example_text)
print(f"Inference Time: {inference_time:.4f} seconds")