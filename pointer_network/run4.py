import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout
from sklearn.model_selection import train_test_split
from PointerNetwork import PointerLSTM

with open('embeddings_100.pkl', 'rb') as file:
    weights = pickle.load(file)

with open('word2idx.pkl', 'rb') as file:
    word2idx = pickle.load(file)

sentences = []
with open('teamstiers3.txt', 'r') as file:
    for line in file:
        sentences.append(line.strip().split())

maxWords = 0
for i in range(len(sentences)):
    if maxWords < len(sentences[i]):
        maxWords = len(sentences[i])

feature_dim = 100
tiers = np.array([0, 0.33, 0.66, 1])
placement_weights = np.array([1, 0.875, 0.75, 0.625, 0.375, 0.25, 0.125, 0])

X = np.zeros((len(sentences), maxWords - 1, feature_dim))
Y = np.zeros((len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        if sentences[i][j] == 'nan':
            X[i][j] = np.zeros((feature_dim,))
        else:
            if j == len(sentences[i]) - 1:
                Y[i] = placement_weights[int(sentences[i][j]) - 1]
            else:
                if str(sentences[i][j]).isdigit():
                    num = int(sentences[i][j])
                    if 1 <= num <= 4:
                        X[i][j].fill(tiers[num - 1])
                    else:
                        X[i][j] = np.zeros((feature_dim,))
                else:
                    X[i][j] = weights[word2idx[sentences[i][j]] - 1]

timesteps = maxWords - 1
input_dim = feature_dim
latent_dim = 100

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(timesteps, input_dim)))
model.add(RepeatVector(timesteps))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(input_dim)))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, X, epochs=10, batch_size=32, verbose=1)

encoder = Model(inputs=model.input, outputs=model.layers[0].output)
X_encoded = encoder.predict(X)

num_teams = 8
num_samples = X_encoded.shape[0] // num_teams

X_pointer = np.zeros((num_samples, num_teams, latent_dim))
Y_pointer = np.zeros((num_samples, num_teams))

for i in range(num_samples):
    X_pointer[i] = X_encoded[i * num_teams:(i + 1) * num_teams]
    sorted_indices = np.argsort(Y[i * num_teams:(i + 1) * num_teams])[::-1]
    Y_pointer[i] = sorted_indices

X_train, X_val, Y_train, Y_val = train_test_split(X_pointer, Y_pointer, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, random_state=42)

hidden_size = 256
seq_len = num_teams
learning_rate = 0.001

print("Building model...")
main_input = Input(shape=(seq_len, latent_dim), name='main_input')

encoder, state_h, state_c = LSTM(hidden_size, return_sequences=True, name="encoder", return_state=True)(main_input)
decoder = PointerLSTM(hidden_size, name="decoder")(encoder, states=[state_h, state_c])

model = Model(main_input, decoder)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=64)

val_loss, val_accuracy = model.evaluate(X_test, Y_test)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

model.save_weights('model_weights_no_tiers_fixed.hdf5')