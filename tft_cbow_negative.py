from tensorflow.python.keras.layers import Embedding, Dot, Input, Flatten
from tensorflow.python.keras.models import Model
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint

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

def generate_training_data(sentences, word2idx, window_size, num_negative_samples):
    X, Y = [], []
    for sentence in sentences:
        sentence = [word2idx[word] for word in sentence]
        for i, target_word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    X.append(target_word)
                    Y.append(sentence[j])
                    negative_samples = np.random.choice(list(word2idx.values()), size=num_negative_samples)
                    for neg_word in negative_samples:
                        X.append(target_word)
                        Y.append(neg_word)
    return np.array(X), np.array(Y)

X, Y = generate_training_data(sentences, word2idx, window_size=3, num_negative_samples=5)
V = len(words)

X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_rest, Y_rest, test_size=0.3, random_state=42)

input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(V, 200, input_length=1, name='embedding')
target_embedding = embedding(input_target)
context_embedding = embedding(input_context)

dot_product = Dot(axes=-1)([target_embedding, context_embedding])
output = Flatten()(dot_product)

model = Model(inputs=[input_target, input_context], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='checkpointy_tft_negative.h5',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True
                             )

model.fit([X_train, Y_train], np.ones(len(X_train)),
          validation_data=([X_val, Y_val], np.ones(len(X_val))),
         epochs=10, batch_size=2048, callbacks=[checkpoint])

loss, accuracy = model.evaluate([X_test, Y_test], np.ones(len(X_test)))

print("Loss:", loss)
print("Accuracy:", accuracy)

word = 'TFT11_Malphite'
target_idx = word2idx[word]
arr = model.predict([np.array([target_idx]*len(words)), np.array(list(word2idx.values()))])
predicted_indices = np.argmax(arr.flatten())

print("Predicted word:", idx2word[predicted_indices])

print("Time taken: " + str(time.time() - start))

pd.DataFrame(embedding.get_weights()).to_csv('negative_embedding.csv', index=False)

tf.keras.backend.clear_session()
