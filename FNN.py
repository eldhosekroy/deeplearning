import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
# --- 1. Data Loading and Preprocessing ---
# Load the MNIST dataset
(X_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize and Flatten the images
X_train =X_train.astype('float32') / 255.0
x_test =x_test.astype('float32') / 255.0
# --- 2. Model Definition (3-Layer FNN) ---
model = Sequential([
    # Layer 1: Input Layer (Implicitly defined by the input to Flatten)
    # The Flatten layer converts the 28x28 image into a 784-dimensional vector.
    Flatten(input_shape=(28, 28)),
    # Layer 2: Hidden Layer (The first computational layer with 128 neurons)
    Dense(128, activation='relu'),
    # Layer 3: Output Layer (10 units for 10 classes, with softmax for probabilities)
    Dense(10, activation='softmax')
])
# --- 3. Compilation and Training ---
model.compile(
    optimizer='adam',
    # 'sparse_categorical_crossentropy' is used because labels are integers (0-9).
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Train the model
model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)
# --- 4. Evaluation ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")