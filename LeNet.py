import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Data Loading and Preprocessing ---
# LeNet was designed for 32x32 input, so we'll resize MNIST's 28x28 images.
INPUT_SHAPE = (32, 32, 1)
NUM_CLASSES = 10

# Load the data
(train_images_orig, train_labels), (test_images_orig, test_labels) = mnist.load_data()

# 1. Reshape and Expand to include channel dimension (1 for grayscale)
train_images = train_images_orig.reshape(-1, 28, 28, 1)
test_images = test_images_orig.reshape(-1, 28, 28, 1)

# 2. Resize images from 28x28 to 32x32 (LeNet-5 standard input)
train_images = tf.image.resize(train_images, INPUT_SHAPE[:2])
test_images = tf.image.resize(test_images, INPUT_SHAPE[:2])

# 3. Normalize pixel values to [0, 1]
train_images = train_images.numpy().astype('float32') / 255.0
test_images = test_images.numpy().astype('float32') / 255.0

# 4. Convert labels to one-hot encoding (optional for sparse_categorical_crossentropy, but common for LeNet context)
# train_labels = to_categorical(train_labels, NUM_CLASSES)
# test_labels = to_categorical(test_labels, NUM_CLASSES)


# --- 2. LeNet-5 Model Definition ---
def build_lenet5(input_shape, num_classes):
    """
    Implements the classic LeNet-5 architecture.
    C1 -> P1 -> C2 -> P2 -> F6 -> Output
    """
    model = Sequential([
        # C1: Convolutional layer (6 filters, 5x5 kernel)
        Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape),
        # P1: Subsampling/Pooling layer (2x2)
        MaxPooling2D(pool_size=(2, 2)),

        # C2: Convolutional layer (16 filters, 5x5 kernel)
        Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
        # P2: Subsampling/Pooling layer (2x2)
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output for the Dense layers
        Flatten(),

        # F6: Fully connected layer (120 units)
        Dense(units=120, activation='tanh'),

        # F7: Fully connected layer (84 units) - Often included in variations
        Dense(units=84, activation='tanh'),

        # Output layer (10 units for 10 classes)
        Dense(units=num_classes, activation='softmax')
    ])
    return model

# --- 3. Instantiate, Compile, and Train ---
lenet_model = build_lenet5(INPUT_SHAPE, NUM_CLASSES)

lenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # Using 'sparse_categorical_crossentropy' since we didn't one-hot encode the labels
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model summary
lenet_model.summary()

print("Starting LeNet-5 model training...")
# Train the model
history = lenet_model.fit(
    x=train_images,
    y=train_labels,
    epochs=3, # A reasonable number of epochs for MNIST
    batch_size=128,
    validation_data=(test_images, test_labels),
    verbose=1
)

# --- 4. Evaluation ---
test_loss, test_acc = lenet_model.evaluate(test_images, test_labels, verbose=0)

print(f"\n✅ LeNet-5 Training Complete.")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
