import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import time
import torch

training_losses = [0.4292, 0.2001, 0.1347, 0.0939, 0.0748]
validation_losses = [0.2624, 0.2311, 0.2455, 0.2674, 0.2530]

def calculate_metrics(model, val_loader):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            _, preds = torch.max(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    
    fpr, tpr, thresholds = roc_curve(all_labels, np.array(all_probs)[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)

    return accuracy, precision, recall, f1, sensitivity, specificity, auc_score, cm, fpr, tpr

accuracy, precision, recall, f1, sensitivity, specificity, auc_score, cm, fpr, tpr = calculate_metrics(model, val_loader)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc_score:.4f}")

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

labels = label_encoder.classes_  
plot_confusion_matrix(cm, labels)

def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(fpr, tpr, auc_score)

def plot_loss_graph(training_losses, validation_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.title("Epoch vs Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_loss_graph(training_losses, validation_losses)

def measure_inference_time(model, text, tokenizer):
    start_time = time.time()
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        max_length=64,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time

text_example = "This is a test text for inference time."
inference_time = measure_inference_time(model, text_example, tokenizer)
print(f"Inference Time: {inference_time:.4f} seconds")
