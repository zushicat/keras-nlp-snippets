import json
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import numpy as np

MODEL_DIR = "models/0_2"

# ***************************************************************
#
# ***************************************************************
def get_super_small_data():
    incoming = {
    "animals": ["a dog barks", "fluffy cat"],
    "mobility": ["i ride a bicycle", "i drive a nice car"],
    "leasure": ["swim", "dance"],
    }

    text_list = []
    label_list = []
    for i, incoming_text_list in enumerate(incoming.values()):
        text_list += incoming_text_list
        label_list += [i] * len(incoming_text_list)

    category_tags = list(incoming.keys())

    return text_list, label_list, category_tags


# ***************************************************************
#
# ***************************************************************
def encode_text(text_list, label_list, category_tags):
    tokenizer = Tokenizer(oov_token='<UNKNOWN>')  # oov_token is optional to explicitly mark token not in word index
    tokenizer.fit_on_texts(text_list)

    vocab_size = len(tokenizer.word_index)+1
    text_sequences = tokenizer.texts_to_sequences(text_list)

    padded_sequences = pad_sequences(text_sequences)
    max_sequence_len = len(padded_sequences[0])  # same as: max([len(x) for x in text_sequences]) or len(max(text_sequences, key=len))

    labels = tf.keras.utils.to_categorical(label_list)
    num_categories = len(category_tags)  #  same as; len(labels[0])

    return tokenizer, vocab_size, padded_sequences, max_sequence_len, labels, num_categories


# ***************************************************************
#
# ***************************************************************
def build_train_model(padded_sequences, labels, vocab_size, max_sequence_len, num_categories):
    num_epochs = 30
    
    input_dim = vocab_size
    output_dim = round(vocab_size ** 0.25)  # rule of thumb of embedding_dimensions according to this stackoverflow answer: https://datascience.stackexchange.com/a/48194
    input_length = max_sequence_len  # (word) length of longest sequence

    X = padded_sequences  # just to stick with conventions
    y = labels

    # *******
    # define model structure
    # hence only sizes (shapes) are given
    # maybe see also: https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
    # *******
    # very simple model a) with embedding layer (needs to be flattend)
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim, output_dim, input_length=input_length))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_categories, activation='softmax'))

    # very simple model b) without embedding layer
    # model = tf.keras.Sequential()
    # model.add(layers.Dense(8, input_shape=(input_length,), activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(num_categories, activation='softmax'))

    # more complex model with more hidden layers (plus: alternative initialization)
    # model = tf.keras.Sequential([
    #     layers.Embedding(input_dim, output_dim, input_length=input_length),
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
        metrics=['accuracy'])
    model.summary()

    # *******
    # train the model on X (values) and Y (labels)
    # history saves callback with metrics
    # *******
    history = model.fit(
        X, 
        y, 
        epochs=num_epochs,
        validation_split=0.1, # or use explicit test data instead: validation_data=(X_test, y_test)
        verbose=0  # use 1 or 2 for output of training metrics
    )  
    
    # plot model metrics
    # plot_history(history)

    return model


def plot_history(history):
    # see also (scroll down): https://realpython.com/python-keras-text-classification/
    plt.style.use('ggplot')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# ***************************************************************
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# ***************************************************************
def save_model(category_tags, tokenizer, model, model_name):
    model_config = model.to_json(indent=2)  # serialize model to JSON
    with open(f"{MODEL_DIR}/{model_name}_config.json", "w") as f:
        f.write(model_config)
    model.save_weights(f"{MODEL_DIR}/{model_name}_weights.h5")  # serialize weights to HDF5

    with open(f"{MODEL_DIR}/{model_name}_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f"{MODEL_DIR}/{model_name}_category_tags.json", "w") as f:
        f.write(json.dumps(category_tags, indent=2, ensure_ascii=False))

def load_model(model_name):
    with open(f"{MODEL_DIR}/{model_name}_config.json") as f:
        model_config = f.read()
    model = model_from_json(model_config)
    model.load_weights(f"{MODEL_DIR}/{model_name}_weights.h5")

    # get max length of sequences (to pad input sequences in prediction accordingly -> same shape as in model)
    max_sequence_len = json.loads(model_config)["config"]["layers"][0]["config"]["input_length"]  # from embedding layer

    with open(f"{MODEL_DIR}/{model_name}_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open(f"{MODEL_DIR}/{model_name}_category_tags.json") as f:
        category_tags = json.load(f)
    
    return category_tags, tokenizer, model, max_sequence_len

# ***************************************************************
#
# ***************************************************************
def predict_text_class(tokenizer, category_tags, model, max_sequence_len):
    sentences = ["i have a cute fluffy cat", "the cat is fluffy", "i like to dance"]

    sequences = tokenizer.texts_to_sequences(sentences)
    decoded_sentences = tokenizer.sequences_to_texts(sequences)

    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len)
    predictions = model.predict(padded_sequences)

    for i, prediction in enumerate(predictions):
        category = category_tags[np.argmax(prediction)]
        print(f"----- sentence {i} -----")
        print(f"decoded text: {decoded_sentences[i]}")
        print(f"sentence: {sentences[i]} | tag: {category}  | prediction values: {prediction} | max prediction index: {np.argmax(prediction)}")



# ********
# build, train and save model
# ********
text_list, label_list, category_tags = get_super_small_data()
tokenizer, vocab_size, padded_sequences, max_sequence_len, labels, num_categories = encode_text(text_list, label_list, category_tags)
model = build_train_model(padded_sequences, labels, vocab_size, max_sequence_len, num_categories)
save_model(category_tags, tokenizer, model, "test")

# ********
# load model and predict class of text(s)
# ********
# category_tags, tokenizer, model, max_sequence_len = load_model("test")
# predict_text_class(tokenizer, category_tags, model, max_sequence_len)
