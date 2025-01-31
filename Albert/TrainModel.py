import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AdamW
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_df = pd.read_csv("/content/drive/MyDrive/datalar/train_data.csv")
test_df = pd.read_csv("/content/drive/MyDrive/datalar/test_data.csv")
label_encoder = LabelEncoder()
train_df['Category'] = label_encoder.fit_transform(train_df['Category'])
test_df['Category'] = label_encoder.transform(test_df['Category'])

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

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

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=len(train_df['Category'].unique()))
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()

EPOCHS = 5

def train_model():
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
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

