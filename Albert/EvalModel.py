def validate_model():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

            

    return total_loss / len(val_loader)

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train_model()
    val_loss = validate_model()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

end_time = time.time()
training_time = end_time - start_time

print(f"Training Time: {training_time:.2f} seconds")

model.save_pretrained("/content/drive/MyDrive/datalar/saved_albert_model")
tokenizer.save_pretrained("/content/drive/MyDrive/datalar/saved_albert_model")