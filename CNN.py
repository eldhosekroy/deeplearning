import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
# ----- Load and prepare the MNIST dataset -----
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape for CNN input (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# ----- Build the CNN model -----
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), # 1st conv layer
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'), # 2nd conv layer
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'), # 3rd conv layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') # Output layer (10 classes)
])
# ----- Compile the model -----
model.compile(optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
# ----- Train the model -----
history = model.fit(x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test))
# ----- Evaluate on test data -----
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
# ----- Plot Accuracy and Loss Graphs -----
plt.figure(figsize=(12,5))
# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("plot3.png")  # give each figure a different name
plt.close()