import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
# --- 1. Data Loading and Preprocessing ---
# LeNet was designed for 32x32 input, so we'll resize MNIST's 28x28 images.
input_shape = (32, 32, 1)
# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 1. Reshape and Expand to include channel dimension (1 for grayscale) and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# 2. Resize images from 28x28 to 32x32 (LeNet-5 standard input)
x_train = tf.image.resize(x_train, input_shape[:2])
x_test = tf.image.resize(x_test, input_shape[:2])
# --- 2. LeNet-5 Model Definition ---
model = Sequential([
    Conv2D(6, (5, 5), activation='tanh', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(120, 'tanh'),
    Dense(84, 'tanh'),
    Dense(10, 'softmax')
])
# --- 3. Instantiate, Compile, and Train ---
model.compile(
    optimizer=Adam(0.001),
    loss=SparseCategoricalCrossentropy,
    metrics=['accuracy']
)
# Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=3,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)
# --- 4. Evaluation ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")