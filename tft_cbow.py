import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Embedding, Lambda, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.backend import mean
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np

start = time.time()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def strip_lines(line):
    line = line.replace(')', '')
    line = line.replace('(', '')
    line = line.replace('\"', '')
    line = line.replace('\n', '')
    line = line.replace(',', '')
    line = line.replace('\'', '')
    line = line.split()
    return line

sentences = []
with open('pary.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        sentences.append(strip_lines(line))

words = set(word for sentence in sentences for word in sentence)
word2idx = {word: i + 1 for i, word in enumerate(words)}
idx2word = {i: word for word, i in word2idx.items()}

sequences = [[word2idx[word] for word in sentence] for sentence in sentences]

V = len(word2idx) + 1
X = []
Y = []
win_size = 1

for seq in sequences:
    for i in range(len(seq)):
        target_word = seq[i]
        context = []
        for j in range(-win_size + i, win_size + 1 + i):
            if j != i:
                if j < 0 or j >= len(seq):
                    context.append(0)
                else:
                    context.append(seq[j])
        X.append(context)
        Y.append(target_word)

X = np.array(X)
Y = to_categorical(Y, num_classes=V)


X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5, random_state=42, shuffle=True)


model = Sequential([
            Embedding(input_dim=V,
                      output_dim=364,
                      input_length=2 * win_size,
                      embeddings_initializer='glorot_uniform'),
            Lambda(lambda x: mean(x, axis=1), output_shape=(364, )),
            Dense(V, activation='softmax', kernel_initializer='glorot_uniform')
        ])

checkpoint = ModelCheckpoint(filepath='checkpointy_tft',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True
                             )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=4096, callbacks=[checkpoint])

loss, accuracy = model.evaluate(X_test, Y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

print("Time taken: " + str(time.time() - start))

tf.keras.backend.clear_session()
