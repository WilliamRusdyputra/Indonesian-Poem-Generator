import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('poem-dataset.txt', 'r') as f:
    corpus = f.read().splitlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index
unique_count = len(word_index)
print(word_index)

input_sequences = []
for line in corpus:
    sequence = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i + 1]
        input_sequences.append(n_gram_sequence)

max_length = max([len(sequence) for sequence in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
input_sequences = np.array(padded_sequences)

train_data, train_labels = input_sequences[:, :-1], input_sequences[:, -1]
train_labels = to_categorical(train_labels, num_classes=unique_count)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(unique_count + 1, 256, input_length=max_length - 1),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(unique_count, activation='softmax')
])

opt = 'adam'
loss = 'categorical_crossentropy'
epochs = 120
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

model.summary()

checkpoint_path = 'model/poem_model.ckpt'

try:
    model.load_weights(checkpoint_path)
except (ValueError, Exception):
    print('Savepoint is not found, begin training...')
    result = model.fit(train_data, train_labels, epochs=epochs, verbose=2)
    model.save_weights(checkpoint_path)
    plt.plot(result.history['accuracy'], label='accuracy')
    plt.legend()
    plt.figure()
    plt.plot(result.history['loss'], label='loss')
    plt.legend()
    plt.show()

text = 'Indonesia'
next_words = 30

for _ in range(next_words):
    tokens = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([tokens], maxlen=max_length - 1, padding='pre')
    predicted = model.predict_classes(padded_sequence)
    output = ''
    for word, index in word_index.items():
        if index == predicted:
            output = word
            break
    text += ' ' + output

text_result = [t if idx % 4 != 0 or idx == 0 else t + '\n' for idx, t in enumerate(text.split())]
print(' ', end='')
for text in text_result:
    print(text + ' ', end='')
