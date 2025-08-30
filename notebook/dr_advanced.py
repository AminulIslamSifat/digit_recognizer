import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import os
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# --- Paths ---
train_image_path = "/home/sifat/AI/data/train_image"
train_label_path = "/home/sifat/AI/data/train_label"
test_image_path = "/home/sifat/AI/data/test_image"
test_label_path = "/home/sifat/AI/data/test_label"
model_path = "/home/sifat/AI/notebook/mnist_cnn.pth"
BOT_TOKEN = "6846587660:AAH9R-W7D3qn98mBfFROiD9vGaixIrwEAno"

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load MNIST functions ---
def load_images(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num, row, col = np.frombuffer(data[:16], dtype=">i4")
    imgs = np.frombuffer(data[16:], dtype=np.uint8).reshape(num, row, col)
    return imgs.astype(np.float32)/255.0

def load_labels(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num = np.frombuffer(data[:8], dtype=">i4")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

# --- Dataset ---
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).unsqueeze(1)  # add channel dim
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# --- Load data ---
X_train = load_images(train_image_path)
Y_train = load_labels(train_label_path)
X_test = load_images(test_image_path)
Y_test = load_labels(test_label_path)

train_dataset = MNISTDataset(X_train, Y_train)
test_dataset = MNISTDataset(X_test, Y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.fc1 = nn.Linear(8*26*26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- Load saved weights if available ---
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded saved model weights.")
else:
    print("No saved weights found, training from scratch.")

# --- Train if needed ---
def train_model(epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")
    torch.save(model.state_dict(), model_path)
    print("Model saved after training.")

# --- Predict ---
def predict_digit(img):
    model.eval()
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()*100, probs.cpu().numpy().flatten()

def preprocess_image(path):
    img = Image.open(path).convert("L").resize((28,28))
    arr = np.array(img).astype(np.float32)/255.0
    return arr

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a handwritten digit image!")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    path = "temp_digit.jpg"
    await file.download_to_drive(path)

    img = preprocess_image(path)
    digit, conf, probs = predict_digit(img)
    prob_str = "\n".join([f"{i}: {p*100:.2f}%" for i,p in enumerate(probs)])
    await update.message.reply_text(f"Prediction: {digit} ({conf:.2f}% confident)\n\nClass probabilities:\n{prob_str}")

    # --- Fine-tuning if label provided ---
    caption = update.message.caption
    if caption is not None and caption.isdigit() and 0 <= int(caption) <= 9:
        label = torch.tensor([int(caption)], dtype=torch.long).to(device)
        model.train()
        optimizer.zero_grad()
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(img_tensor)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), model_path)
        await update.message.reply_text("Model fine-tuned with your input!")

# --- Run Bot ---
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, handle_image))
app.run_polling()
