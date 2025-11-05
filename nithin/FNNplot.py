import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# ------------------------------
# Load and prepare the MNIST dataset
# ------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# ------------------------------
# Define multiple optimizers
# ------------------------------
optimizers = {
    "Adam": Adam(),
    "SGD": SGD(),
    "SGD with Momentum": SGD(learning_rate=0.01, momentum=0.9)
}

history_dict = {}

# ------------------------------
# Train and evaluate each optimizer
# ------------------------------
for name, opt_instance in optimizers.items():
    print(f"\n🔹 Using optimizer: {name}")

    # Build the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=opt_instance,
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

    history_dict[name] = history

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test accuracy with {name}: {test_acc:.4f}")
    print(f"🧮 Final validation loss: {test_loss:.4f}")

# ------------------------------
# Plot 1: Validation Accuracy Comparison (across optimizers)
# ------------------------------
plt.figure(figsize=(10, 5))
for name, history in history_dict.items():
    plt.plot(history.history['val_sparse_categorical_accuracy'], label=f'{name}')
plt.title('Validation Accuracy Comparison (Optimizers)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Plot 2: Training vs Validation Accuracy for each optimizer
# ------------------------------
plt.figure(figsize=(12, 6))
for name, history in history_dict.items():
    plt.plot(history.history['sparse_categorical_accuracy'], linestyle='--', label=f'{name} - Train Acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label=f'{name} - Val Acc')
plt.title('Training vs Validation Accuracy (All Optimizers)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Plot 3: Training vs Validation Loss for each optimizer
# ------------------------------
plt.figure(figsize=(12, 6))
for name, history in history_dict.items():
    plt.plot(history.history['loss'], linestyle='--', label=f'{name} - Train Loss')
    plt.plot(history.history['val_loss'], label=f'{name} - Val Loss')
plt.title('Training vs Validation Loss (All Optimizers)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
