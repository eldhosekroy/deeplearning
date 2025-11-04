#import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
num_words = 10000
maxlen = 200 # truncate or pad reviews to 200 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print("Training samples:", len(x_train))
print("Test samples:", len(x_test))
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
x_train, y_train,
epochs=3,
batch_size=64,
validation_split=0.2,
verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")