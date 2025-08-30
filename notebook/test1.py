import numpy as np
import os

# --- Paths ---
train_image_path = "/home/sifat/AI/data/train_image"
train_label_path = "/home/sifat/AI/data/train_label"
test_image_path = "/home/sifat/AI/data/test_image"
test_label_path = "/home/sifat/AI/data/test_label"

# --- Load MNIST functions ---
def load_images(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num, row, col = np.frombuffer(data[:16], dtype=">i4")
    if magic != 2051:
        raise ValueError("Invalid image file")
    imgs = np.frombuffer(data[16:], dtype=np.uint8).reshape(num, row*col)
    return imgs.astype(np.float32)/255.0

def load_labels(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, num = np.frombuffer(data[:8], dtype=">i4")
    if magic != 2049:
        raise ValueError("Invalid label file")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

# --- Load raw MNIST data directly ---
print("Loading MNIST data...")
X_train = load_images(train_image_path)
Y_train = one_hot(load_labels(train_label_path))
X_test = load_images(test_image_path)
Y_test = one_hot(load_labels(test_label_path))

# --- CNN layers ---
def conv2d(X, W, b, stride=1):
    n_filters, f_h, f_w = W.shape
    H, W_in = X.shape
    out_h = H - f_h + 1
    out_w = W_in - f_w + 1
    out = np.zeros((n_filters, out_h, out_w))
    for k in range(n_filters):
        for i in range(out_h):
            for j in range(out_w):
                out[k,i,j] = np.sum(X[i:i+f_h,j:j+f_w]*W[k]) + b[k]
    return out

def relu(X):
    return np.maximum(0, X)

def relu_grad(X):
    return (X > 0).astype(np.float32)

def maxpool2d(X, size=2, stride=2):
    n_filters, H, W_in = X.shape
    out_h = (H - size)//stride +1
    out_w = (W_in - size)//stride +1
    out = np.zeros((n_filters, out_h, out_w))
    for k in range(n_filters):
        for i in range(out_h):
            for j in range(out_w):
                out[k,i,j] = np.max(X[k,i*stride:i*stride+size,j*stride:j*stride+size])
    return out

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e/np.sum(e)

# --- Initialize or load CNN weights ---
if os.path.exists("conv1_W.npy") and os.path.exists("conv1_b.npy") \
   and os.path.exists("fc_W.npy") and os.path.exists("fc_b.npy"):
    print("Loading existing weights...")
    conv1_W = np.load("conv1_W.npy")
    conv1_b = np.load("conv1_b.npy")
    fc_W = np.load("fc_W.npy")
    fc_b = np.load("fc_b.npy")
else:
    print("Initializing new weights...")
    conv1_W = np.random.randn(8,3,3)*0.1
    conv1_b = np.zeros(8)
    fc_W = np.random.randn(8*13*13,10)*0.1
    fc_b = np.zeros(10)

# --- Forward pass ---
def forward(img):
    conv_out = conv2d(img, conv1_W, conv1_b)
    conv_out_relu = relu(conv_out)
    pooled = maxpool2d(conv_out_relu)
    flat = pooled.flatten().reshape(1,-1)
    logits = flat @ fc_W + fc_b
    probs = softmax(logits)
    cache = (conv_out, conv_out_relu, pooled, flat)
    return probs, cache

# --- Training hyperparameters ---
epochs = 3
batch_size = 32
lr = 0.01

for e in range(epochs):
    idx = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[idx]
    Y_train_shuffled = Y_train[idx]
    batch_loss = 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        Y_batch = Y_train_shuffled[i:i+batch_size]

        d_fc_W = np.zeros_like(fc_W)
        d_fc_b = np.zeros_like(fc_b)
        d_conv_W = np.zeros_like(conv1_W)
        d_conv_b = np.zeros_like(conv1_b)

        for j in range(X_batch.shape[0]):
            img = X_batch[j].reshape(28,28)  # reshape to 2D
            y_true = Y_batch[j].reshape(1,-1)
            probs, cache = forward(img)
            conv_out, conv_out_relu, pooled, flat = cache
            batch_loss += -np.sum(y_true*np.log(probs+1e-15))

            dlogits = probs - y_true
            d_fc_W += flat.T @ dlogits
            d_fc_b += dlogits.flatten()

            d_flat = dlogits @ fc_W.T
            d_pooled = d_flat.reshape(pooled.shape)

            d_conv_relu = np.zeros_like(conv_out_relu)
            n_filters, out_h, out_w = pooled.shape
            for k in range(n_filters):
                for i_pool in range(out_h):
                    for j_pool in range(out_w):
                        patch = conv_out_relu[k,i_pool*2:i_pool*2+2,j_pool*2:j_pool*2+2]
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        d_conv_relu[k,i_pool*2 + max_idx[0], j_pool*2 + max_idx[1]] = d_pooled[k,i_pool,j_pool]

            d_conv = d_conv_relu * relu_grad(conv_out)

            for k in range(conv1_W.shape[0]):
                for i_f in range(conv1_W.shape[1]):
                    for j_f in range(conv1_W.shape[2]):
                        d_conv_W[k,i_f,j_f] += np.sum(d_conv[k] * img[i_f:i_f + d_conv.shape[1], j_f:j_f + d_conv.shape[2]])
                d_conv_b[k] += np.sum(d_conv[k])

        conv1_W -= lr * d_conv_W / X_batch.shape[0]
        conv1_b -= lr * d_conv_b / X_batch.shape[0]
        fc_W -= lr * d_fc_W / X_batch.shape[0]
        fc_b -= lr * d_fc_b / X_batch.shape[0]

    batch_loss /= X_train.shape[0]

    correct = 0
    for j in range(X_train.shape[0]):
        img = X_train[j].reshape(28,28)
        probs, _ = forward(img)
        pred = np.argmax(probs)
        label = np.argmax(Y_train[j])
        if pred == label:
            correct += 1
    acc = correct / X_train.shape[0]
    print(f"Epoch {e+1}/{epochs}, Loss: {batch_loss:.4f}, Accuracy: {acc*100:.2f}%")

# --- Save weights ---
np.save("conv1_W.npy", conv1_W)
np.save("conv1_b.npy", conv1_b)
np.save("fc_W.npy", fc_W)
np.save("fc_b.npy", fc_b)

# --- Test accuracy ---
correct = 0
for j in range(X_test.shape[0]):
    img = X_test[j].reshape(28,28)
    probs, _ = forward(img)
    pred = np.argmax(probs)
    label = np.argmax(Y_test[j])
    if pred == label:
        correct += 1
test_acc = correct / X_test.shape[0]
print(f"Test accuracy: {test_acc*100:.2f}%")
