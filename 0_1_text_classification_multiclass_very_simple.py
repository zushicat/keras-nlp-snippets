# https://chatbotslife.com/how-to-build-a-chatbot-from-zero-a0ebb186b070
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np


# ***************************************************************
#
# get data
#
# ***************************************************************
incoming = {
  "animals": ["a dog barks", "fluffy cat"],
  "mobility": ["i ride a bicycle", "i drive a car"],
  "leasure": ["swim", "dance"],
}

text_list = []
label_list = []
for i, incoming_text_list in enumerate(incoming.values()):
    text_list += incoming_text_list
    label_list += [i] * len(incoming_text_list)

category_tags = list(incoming.keys())


# ***************************************************************
#
# encode text / for word embedding (embedding layer)
# https://raghakot.github.io/keras-text/keras_text.processing/
# https://stackoverflow.com/a/53921105
#
# ***************************************************************
'''
1) text_list -> 2) text_sequences -> 3) padded_sequences, i.e.:
1-  {'<UNKNOWN>': 1, 'a': 2, 'i': 3, 'dog': 4, 'barks': 5, 'fluffy': 6, 'cat': 7, 'ride': 8, 'bicycle': 9, 'drive': 10, 'car': 11, 'swim': 12, 'dance': 13}
2-  [[2, 4, 5], [6, 7], [3, 8, 2, 9], [3, 10, 2, 11], [12], [13]]  # replace <str> sentences with list of word indixes
3-  [                 # pad all sentences to max. sequence length
      [ 0  2  4  5]   # "a dog barks"
      [ 0  0  6  7]   # "fluffy cat"
      [ 3  8  2  9]   # "i ride a bicycle"
      [ 3 10  2 11]   # "i drive a car"
      [ 0  0  0 12]   # "swim"
      [ 0  0  0 13]   # "dance"
    ]   
'''
tokenizer = Tokenizer(oov_token='<UNKNOWN>')  # oov_token is optional to explicitly mark token not in word index
tokenizer.fit_on_texts(text_list)

vocab_size = len(tokenizer.word_index)+1
text_sequences = tokenizer.texts_to_sequences(text_list)

padded_sequences = pad_sequences(text_sequences)
max_sequence_len = len(padded_sequences[0])  # same as: max([len(x) for x in text_sequences]) or len(max(text_sequences, key=len))

labels = tf.keras.utils.to_categorical(label_list)
num_categories = len(category_tags)  #  same as; len(labels[0])


# ***************************************************************
#
# model
#
# ***************************************************************
num_epochs = 50
X = padded_sequences  # just to stick with conventions
Y = labels

# *******
# define model structure
# hence only sizes (shapes) are given
# *******
# very simple model
model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=max_sequence_len))  # input_dim, output_dim, input_length
model.add(layers.Flatten())
model.add(layers.Dense(num_categories, activation='sigmoid'))

# more layers (plus: alternative initialization)
# model = tf.keras.Sequential([
#     layers.Embedding(vocab_size, 16, input_length=max_sequence_len),  # input_dim, output_dim, input_length
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_categories, activation='softmax')
# ])

# *******
# define matrix opeations ("behavior") on model structure
# *******
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
model.summary()

# *******
# train the model on X (values) and Y (labels)
# *******
history = model.fit(X, Y, epochs=num_epochs, verbose=0)  # use verbose= 1 or 2 for output on training


# ***************************************************************
#
# predict
#
# ***************************************************************
sentences = ["i have a cute fluffy cat", "the cat is fluffy", "i like to dance"]

sequences = tokenizer.texts_to_sequences(sentences)
decoded_sentences = tokenizer.sequences_to_texts(sequences)

padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len)
predictions = model.predict(padded_sequences)

for i, prediction in enumerate(predictions):
  category = category_tags[np.argmax(prediction)]
  print(f"----- sekntence {i} -----")
  print(f"decoded text: {decoded_sentences[i]}")
  print(f"sentence: {sentences[i]} | tag: {category}  | prediction values: {prediction} | max prediction index: {np.argmax(prediction)}")
