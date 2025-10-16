import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# --- 1. Data Loading and Preprocessing ---
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and Flatten the images
# The Flatten layer will handle converting the 28x28 images into a 784-element vector.
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# --- 2. Model Definition (3-Layer FNN) ---
model = Sequential([
    # Layer 1: Input Layer (Implicitly defined by the input to Flatten)
    # The Flatten layer converts the 28x28 image into a 784-dimensional vector.
    Flatten(input_shape=(28, 28)),

    # Layer 2: Hidden Layer (The first computational layer with 128 neurons)
    Dense(units=128, activation='relu'),

    # Layer 3: Output Layer (10 units for 10 classes, with softmax for probabilities)
    Dense(units=10, activation='softmax')
])

# --- 3. Compilation and Training ---
model.compile(
    optimizer='adam',
    # 'sparse_categorical_crossentropy' is used because labels are integers (0-9).
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model architecture
print(model.summary())

print("\nStarting 3-Layer FNN training...")

# Train the model
model.fit(
    x=train_images,
    y=train_labels,
    epochs=3,
    batch_size=32,
    validation_data=(test_images, test_labels),
    verbose=1
)

# --- 4. Evaluation ---
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

print(f"\n✅ Training Complete.")
print(f"Test Accuracy: {test_acc*100:.2f}%")
