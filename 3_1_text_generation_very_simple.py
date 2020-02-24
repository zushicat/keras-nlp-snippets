# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
'''
#Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


def get_data():
    with open("../../data/chefkoch_recipe_texts/dessert_instructions_short.txt") as f:
        text = f.read().lower()

    # get rid of all linebreaks and double spaces
    text = text.splitlines()
    # text = text[:int(len(text)/5)]  # make text even shorter
    text = [x for x in text if len(x) > 0]
    text = " ".join(text)

    return text


def text_encoding(text):
    '''
    1) get unique chars and assign index to chars and vice versa
    2) Scan over text with steps, make sequences of maxlen and save next char at same i in different list.
    Example with steps=1, maxlen=5:
    text =              "abcd ef ghij"
    sentences (X)   =   ['abcd ', 'bcd e', 'cd ef', 'd ef ', ' ef g', 'ef gh', 'f ghi']
    next_chars (y)  =   ['e', 'f', ' ', 'g', 'h', 'i', 'j']

    3) one-hot enccode X and y
    X.shape:    (7, 5, 11)
    y.shape:    (7, 11)
    7: num sequences
    5: (max) len sequences (only X has sequences)
    11: vocab size

    (This is for demonstration only and will be replaces by tokenizer in later scripts.)
    '''
    # 1
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print('vocab_size:', vocab_size)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # 2
    max_len = 100  # length of each sequence
    sequences = []  # seqence in -> X
    next_chars = []  # sequence out -> y
    for i in range(0, len(text) - max_len, 1):
        sequences.append(text[i:i + max_len])
        next_chars.append(text[i + max_len])
    print('num sequences:', len(sequences))

    # 3
    print('Vectorization...')
    X = np.zeros((len(sequences), max_len, vocab_size))
    y = np.zeros((len(sequences), vocab_size))
    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            X[i, j, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1
    
    return X, y, char_to_int, int_to_char, sequences, max_len, vocab_size


text = get_data()
X, y, char_to_int, int_to_char, sequences, max_len, vocab_size = text_encoding(text)

# *************************************************
# build very simple model with single LSTM
# *************************************************
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(
    X, y,
    batch_size=128,
    epochs=200
)

# *************************************************
# generate seed and predict / generate text
# *************************************************
for _ in range(20):
    start = np.random.randint(0, len(sequences)-1)
    random_seed_text = sequences[start]
    text_len = 200
    generated_text = []

    for _ in range(text_len):
        X_pred = np.zeros((1, max_len, vocab_size))
        for i, char in enumerate(random_seed_text):
            X_pred[0, i, char_to_int[char]] = 1.

        prediction = model.predict(X_pred, verbose=0)
        index = np.argmax(prediction)
        next_char = int_to_char[index]

        generated_text.append(next_char)
        
        random_seed_text = random_seed_text[1:] + next_char

    generated_text = f"{random_seed_text}{''.join(generated_text)}"
    print(generated_text)
    print("--------")

