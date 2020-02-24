# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import json
import numpy as np
import pickle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, model_from_json, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant


MODEL_DIR = "models/2_2"
DATA_DIR = "../../data/recipe_texts/dessert_recipe_names.txt"
EMBEDDING_DIM = 100
EMBEDDING_FILEPATH = f"../../data/glove/german_vectors_{EMBEDDING_DIM}_char.txt"

# ***************************************************
#
# ***************************************************
def get_data():
    with open(DATA_DIR) as f:
        text = f.read().lower()

    text = text.splitlines()
    
    text = text[:int(len(text)/5)]  # make very short text for tests
    text = [x for x in text if len(x) > 0]
    text = "\n".join(text)
    
    return text


# ***************************************************
# create char embedding matrix
# https://minimaxir.com/2017/04/char-embeddings/
#
# generally:
# https://keras.io/examples/pretrained_word_embeddings/
# Creating embedding matrix can take a while, hence implemented load/save in test()
# ***************************************************
def create_word_embedding_glove(tokenizer, vocab_size):
    embeddings_index = {}
    with open(EMBEDDING_FILEPATH) as f:
        for line in f:
            try:
                line = line.strip().split(" ")
                word = line[:1][0]

                coefs = np.array([float(val) for val in line[1:]])
                embeddings_index[word] = coefs
            except Exception as e:
                print(e)
                continue
            
    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    for char, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    with open(f"{MODEL_DIR}/embedding_matrix.pickle", "wb") as f:
        pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_word_embedding_glove():
    with open(f"{MODEL_DIR}/embedding_matrix.pickle", "rb") as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix


# ***************************************************
#
# ***************************************************
def text_encoder(text, max_len_sequence, vocab_size, tokenizer, one_hot_enc_x=False):
    sequences = []  # seqence in -> X
    next_chars = []  # sequence out -> y
    for i in range(0, len(text) - max_len_sequence, 1):
        sequences.append(text[i:i + max_len_sequence])
        next_chars.append(text[i + max_len_sequence])
    
    print('num sequences:', len(sequences))

    # sequences
    sequences_indexed = tokenizer.texts_to_sequences(sequences)
    next_chars_indexed = tokenizer.texts_to_sequences(next_chars)
    
    # padding
    padded_sequences = pad_sequences(sequences_indexed, maxlen=max_len_sequence)
    padded_next_chars = pad_sequences(next_chars_indexed)

    # one-hot encoded
    one_hot_sequences = to_categorical(padded_sequences, num_classes=vocab_size)
    one_hot_next_chars = to_categorical(padded_next_chars, num_classes=vocab_size)
    
    if one_hot_enc_x is False:
        return padded_sequences, one_hot_next_chars
    else:
        return one_hot_sequences, one_hot_next_chars


# *************************************************
# build very simple model with single LSTM
# *************************************************
def build_model(vocab_size, max_len_sequence, embedding_matrix, hidden_layer, dropout):
    encoder_embedding_layer = Embedding(
        input_dim = vocab_size, 
        output_dim = embedding_matrix.shape[1], # dimension of embedding matrix i.e. 100
        name = "encoder_embedding",
        input_length = max_len_sequence,
        embeddings_initializer=Constant(embedding_matrix),
        trainable = True,
        mask_zero="True"
    )
    
    model = Sequential()
    model.add(encoder_embedding_layer)
    model.add(LSTM(hidden_layer, input_shape=(max_len_sequence, vocab_size), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='softmax'))
    
    return model


# ***************************************************
# save / load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# ***************************************************
def save_lstm_model(tokenizer, model, model_name):
    model_config = model.to_json(indent=2)  # serialize model to JSON (str)
    with open(f"{MODEL_DIR}/{model_name}_config.json", "w") as f:
        f.write(model_config)
    
    model.save_weights(f"{MODEL_DIR}/{model_name}_weights.h5")  # serialize weights to HDF5

    with open(f"{MODEL_DIR}/{model_name}_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_lstm_model(model_name):
    with open(f"{MODEL_DIR}/{model_name}_config.json") as f:
        model_config = f.read()
    model = model_from_json(model_config)
    model.load_weights(f"{MODEL_DIR}/{model_name}_weights.h5")

    with open(f"{MODEL_DIR}/{model_name}_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    
    return tokenizer, model


# ***************************************************
#
# ***************************************************
def train(model_name="test", epochs=10):
    text = get_data()
    
    # *****
    # fit tokenize on all text
    tokenizer = Tokenizer(char_level=True)  # see also: https://minimaxir.com/2017/04/char-embeddings/
    tokenizer.fit_on_texts(text)
    
    vocab_size = len(tokenizer.word_index)+1
    # max_len_sequence = max([len(x) for x in text]) # int(np.max([len(x) for x in text.split("\n")]))
    max_len_sequence = 40
   
    # *****
    # create new embedding matrix if change input data
    create_word_embedding_glove(tokenizer, vocab_size)
    embedding_matrix = load_word_embedding_glove()
    
    # *****
    #
    X, y = text_encoder(text, max_len_sequence, vocab_size, tokenizer, False)
    
    # *****
    #
    hidden_layer = 64
    dropout = 0.2
    model = build_model(vocab_size, max_len_sequence, embedding_matrix, hidden_layer, dropout)
    model.compile(loss='categorical_crossentropy', optimizer="adam")  # rmsprop
    model.fit(X, y, batch_size=128, epochs=epochs)

    save_lstm_model(tokenizer, model, model_name)

def continue_train(old_model_name="test", new_model_name="test", epochs=10):
    text = get_data()

    # *****
    # load
    tokenizer, model = load_lstm_model(old_model_name)

    max_len_sequence = model.input.shape[1]
    vocab_size = len(tokenizer.word_index)+1
    
    # *****
    #
    X, y = text_encoder(text, max_len_sequence, vocab_size, tokenizer, False)

    # *****
    #
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    model.fit(X, y, batch_size=128, epochs=epochs)

    save_lstm_model(tokenizer, model, new_model_name)

# *************************************************
# generate seed and predict / generate text
# *************************************************
def generate(model_name="test"):
    tokenizer, model = load_lstm_model(model_name)
    # model.summary()
    # print(tokenizer.word_index)

    max_len_sequence = model.input.shape[1]
    vocab_size = len(tokenizer.word_index)+1

    seeds = ["Litschis mit"]
    for i in range(len(seeds)):
        seed_str = seeds[i]
        generated_str = seed_str
        
        x = tokenizer.texts_to_sequences([seed_str])
        x = pad_sequences(x, maxlen=max_len_sequence)
        
        for j in range(1000): #(max_len_sequence-1):
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)

            next_char = tokenizer.sequences_to_texts([[index]])[0]
            if next_char == "\n":
                break
            generated_str += next_char

            tmp = [n for n in list(x[0]) if n != 0][1:]
            tmp += [index]
            x = pad_sequences([tmp], maxlen=max_len_sequence)
            
        print(generated_str)
        print("--------")
            


# train("dessert_names/dessert_100", 100)
# continue_train("dessert_names/dessert_100", "dessert_names/dessert_150", 50)
generate("dessert_names/dessert_100")
