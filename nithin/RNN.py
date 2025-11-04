import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt
# --- Load & preprocess data ---
num_words, maxlen = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test  = pad_sequences(x_test, maxlen=maxlen)
# --- Build model ---
model = Sequential([
    Embedding(num_words, 128, input_length=maxlen),
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])
# --- Compile & train ---
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
# --- Evaluate ---
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
# --- Plot Accuracy & Loss ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("plot_rnn.png")
plt.close()