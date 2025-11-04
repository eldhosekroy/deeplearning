import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

optimizers = {
    "Adam" : Adam(),
    "SGD" : SGD(),
    "SGD with Momentum" : SGD(learning_rate=0.01, momentum=0.9)
}
for name,opt_instance in optimizers.items():
    print(f"\n using optimizer : {name}")

    # Build the model
    model = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')
    ])

      # Compile the model
    model.compile(optimizer= opt_instance,
       loss=SparseCategoricalCrossentropy(),
       metrics=[SparseCategoricalAccuracy()])

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc}')