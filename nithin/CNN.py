import tensorflow as tf
import matplotlib.pyplot as plt

# Load and normalize dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train =x_train.reshape(-1,28,28,1) / 255.0
x_test =x_test.reshape(-1,28,28,1) / 255.0  # Add channel dim

# Build a simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile & train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# Evaluate
print("\nTest accuracy:", model.evaluate(x_test, y_test, verbose=1)[1])

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend(); plt.grid(True)
plt.savefig("plot1.png")
plt.close()
#plt.show()