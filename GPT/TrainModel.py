import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2ForSequenceClassification
from transformers import AdamW
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("/content/train_data.csv")
test_df = pd.read_csv("/content/test_data.csv")

label_encoder = LabelEncoder()
train_df['Category'] = label_encoder.fit_transform(train_df['Category'])
test_df['Category'] = label_encoder.transform(test_df['Category'])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 

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

train_dataset = TextDataset(
    train_df['Content'].values,
    train_df['Category'].values,
    tokenizer,
    MAX_LEN
)

test_dataset = TextDataset(
    test_df['Content'].values,
    test_df['Category'].values,
    tokenizer,
    MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=len(train_df['Category'].unique()))
model.config.pad_token_id = model.config.eos_token_id  
model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()


def train_model():
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_model():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(val_loader)

import time  


EPOCHS = 5
total_training_time = 0  

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

   
    start_time = time.time()

    train_loss = train_model()
    val_loss = validate_model()

   
    end_time = time.time()
    epoch_time = end_time - start_time
    total_training_time += epoch_time

    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Epoch {epoch + 1} Training Time: {epoch_time:.2f} seconds\n")


print(f"Total Training Time: {total_training_time:.2f} seconds")



