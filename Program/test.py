import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os

# ---- Paths ----
train_image_path = "/home/sifat/AI/data/train_image"
train_label_path = "/home/sifat/AI/data/train_label"
test_image_path = "/home/sifat/AI/data/test_image"
test_label_path = "/home/sifat/AI/data/test_label"
BOT_TOKEN = "6846587660:AAH9R-W7D3qn98mBfFROiD9vGaixIrwEAno"

# ---- Load MNIST ----
def load_images(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num, row, col = np.frombuffer(data[:16], dtype=">i4")
    if magic != 2051: raise ValueError("Invalid image file")
    imgs = np.frombuffer(data[16:], dtype=np.uint8).reshape(num, row*col)
    return imgs

def load_labels(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num = np.frombuffer(data[:8], dtype=">i4")
    if magic != 2049: raise ValueError("Invalid label file")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

X_train_raw, Y_train_raw = load_images(train_image_path), load_labels(train_label_path)
X_test_raw, Y_test_raw = load_images(test_image_path), load_labels(test_label_path)

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

X_train, X_test = X_train_raw.astype(np.float32)/255.0, X_test_raw.astype(np.float32)/255.0
Y_train, Y_test = one_hot(Y_train_raw), one_hot(Y_test_raw)

# ---- Network parameters ----
layer_sizes = [784, 728, 512, 256, 128, 64, 32, 10]
learning_rate = 0.03

# Initialize or load weights
weights = []
biases = []
for i in range(len(layer_sizes)-1):
    w_file = f"Program/w{i+1}.npy"
    b_file = f"Program/b{i+1}.npy"
    if os.path.exists(w_file):
        weights.append(np.load(w_file))
        biases.append(np.load(b_file))
    else:
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i]))
        biases.append(np.zeros(layer_sizes[i+1]))

# ---- Activation functions ----
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

# ---- Forward ----
def forward(x):
    activations = [x]
    pre_activations = []
    for i in range(len(weights)-1):
        z = activations[-1] @ weights[i] + biases[i]
        a = relu(z)
        pre_activations.append(z)
        activations.append(a)
    # Output layer
    z = activations[-1] @ weights[-1] + biases[-1]
    a = softmax(z)
    pre_activations.append(z)
    activations.append(a)
    return pre_activations, activations

# ---- Loss ----
def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))

# ---- Backprop ----
def train_step(x, y, lr=learning_rate):
    global weights, biases
    pre_act, act = forward(x)
    loss = cross_entropy(y, act[-1])

    # Gradients
    delta = (act[-1] - y) / x.shape[0]
    grad_w = []
    grad_b = []

    for i in reversed(range(len(weights))):
        grad_w.insert(0, act[i].T @ delta)
        grad_b.insert(0, np.sum(delta, axis=0))
        if i != 0:
            delta = (delta @ weights[i].T) * relu_deriv(pre_act[i-1])

    # Update weights
    for i in range(len(weights)):
        weights[i] -= lr * grad_w[i]
        biases[i] -= lr * grad_b[i]

    return loss

# ---- Accuracy ----
def accuracy(x, y):
    _, act = forward(x)
    preds = np.argmax(act[-1], axis=1)
    labels = np.argmax(y, axis=1)
    return np.mean(preds == labels)

# ---- Save model ----
def save_model():
    for i in range(len(weights)):
        np.save(f"Program/w{i+1}.npy", weights[i])
        np.save(f"Program/b{i+1}.npy", biases[i])

# ---- Preprocess image ----
def preprocess_image(path):
    img = Image.open(path).convert("L").resize((28,28))
    arr = np.array(img).astype(np.float32)/255.0
    return arr.flatten().reshape(1,-1)

def predict_digit(img):
    _, act = forward(img)
    probs = act[-1]
    d = np.argmax(probs)
    conf = probs[0,d]*100
    return d, conf, probs[0]

# ---- Telegram Handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a digit image, and I’ll predict it!")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    path = "temp_digit.jpg"
    await file.download_to_drive(path)

    img = preprocess_image(path)
    digit, conf, probs = predict_digit(img)
    prob_str = "\n".join([f"{i}: {p*100:.2f}%" for i,p in enumerate(probs)])
    await update.message.reply_text(f"Prediction: {digit} ({conf:.2f}% confident)\n\nClass probabilities:\n{prob_str}")

    # If captioned with true label, learn from it
    if update.message.caption and update.message.caption.isdigit():
        label = int(update.message.caption)
        y_true = one_hot(np.array([label]))
        loss = train_step(img, y_true, lr=0.01)
        save_model()
        await update.message.reply_text(f"Trained on your input! Loss: {loss:.4f}")

# ---- Run Bot ----
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, handle_image))
app.run_polling()
