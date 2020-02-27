# https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms

# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import json
import numpy as np
import pickle
from random import shuffle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, model_from_json, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant


MODEL_DIR = "models/2_3"
DATA_DIR = "../../data/recipe_texts/dessert_instructions.txt"
EMBEDDING_DIM = 50
EMBEDDING_FILEPATH = f"../../data/glove/german_vectors_{EMBEDDING_DIM}.txt"

# ***************************************************
#
# ***************************************************
def get_data():
    with open(DATA_DIR) as f:
        text = f.read().lower()

    # some minor text cleaning
    text = text.replace("-", " ").replace("&", " und ").replace("(", "").replace(")", "").replace("/", " ")
    
    text = text.splitlines()
    text = [f"{x.strip()} <EOS>" for x in text if len(x) > 0]  # disregard empty lines and add EOS tag
    text = text[:100]  # make shorter text for tests
    # shuffle(text)
    print("----->", len(text))
    
    return text


# ***************************************************
# create word embedding matrix
# https://keras.io/examples/pretrained_word_embeddings/
# Creating embedding matrix can take a while, hence implemented load/save in test()
# ***************************************************
def create_word_embedding_glove(tokenizer, vocab_size):
    embeddings_index = {}
    with open(EMBEDDING_FILEPATH) as f:
        for line in f:
            try:
                line = line.strip().split("\t")
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
# https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
# ***************************************************
def text_encoder(text, vocab_size, tokenizer):
    def generate_padded_sequences(input_sequences):
        max_sequence_len = max([len(x) for x in input_sequences])  # len of words, not chars
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]  # y is the last token from input
        label = to_categorical(label, num_classes=vocab_size)  # y is one-hot encoded
        
        return predictors, label, max_sequence_len
        
    input_sequences = []
    '''
    from kaggle script:
    expand sequence until max len line: [a b c d] -> [a b] [a b c] [a b c d]
    then get X, y: a -> b | a b -> c | a b c -> d
    '''
    for line in text:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    '''
    To be tested:
    scan over text with max len: [a b c d e f] -> [a b c] [b c d] [c d e] [d e f]
    then get X, y: a b -> c | b c -> d | c d -> e | d e -> f
    '''
    # max_sequence_len = 20  # len of words, not chars
    # for line in text:
    #     token_list = tokenizer.texts_to_sequences([line])[0]
    #     for i in range(0, len(token_list)):
    #         n_gram_sequence = token_list[i:i+max_sequence_len]
    #         input_sequences.append(n_gram_sequence)
    
    return generate_padded_sequences(input_sequences)  # X, y, max_len_seq


# *************************************************
# 
# *************************************************
def build_model(vocab_size, max_len_sequence, embedding_matrix, hidden_layer, dropout):
    max_len_sequence = max_len_sequence - 1  # last token is y

    encoder_embedding_layer = Embedding(
        input_dim = vocab_size, 
        output_dim = embedding_matrix.shape[1], # dimension of embedding matrix i.e. 100
        name = "encoder_embedding",
        input_length = max_len_sequence,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False  # keep the embeddings fixed or not?
    )
    
    model = Sequential()
    model.add(encoder_embedding_layer)
    model.add(LSTM(hidden_layer, input_shape=(max_len_sequence, vocab_size)))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.summary()
    return model


# ***************************************************
# save / load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# ***************************************************
def save_lstm_model(tokenizer, model, model_name):
    '''
    Don't use save_weights() and load_weights() along with Adam.
    These functions save only the model weights, but not the optimizer.
    should be changed to model.save() / load_model()
    see: https://stackoverflow.com/questions/45424683/how-to-continue-training-for-a-saved-and-then-loaded-keras-model
    '''
    model.save(f"{MODEL_DIR}/{model_name}.h5")

    with open(f"{MODEL_DIR}/{model_name}_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_lstm_model(model_name):
    '''
    Don't use save_weights() and load_weights() along with Adam.
    These functions save only the model weights, but not the optimizer.
    should be changed to model.save() / load_model()
    see: https://stackoverflow.com/questions/45424683/how-to-continue-training-for-a-saved-and-then-loaded-keras-model
    '''
    model = load_model(f"{MODEL_DIR}/{model_name}.h5")

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
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    
    vocab_size = len(tokenizer.word_index)+1
    
    # *****
    # create new embedding matrix if change input data
    # create_word_embedding_glove(tokenizer, vocab_size)  # takes a while, hence create once on given input
    embedding_matrix = load_word_embedding_glove()
    
    # *****
    #
    X, y, max_len_sequence = text_encoder(text, vocab_size, tokenizer)
    
    # *****
    #
    hidden_layer = 64
    dropout = 0.2
    model = build_model(vocab_size, max_len_sequence, embedding_matrix, hidden_layer, dropout)
    
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    
    # *****
    # reduces the learning rate once the validation loss hasn't improved for a given number of epochs
    # https://keras.io/callbacks/#reducelronplateau
    model.fit(
        X, 
        y, 
        batch_size=128, 
        epochs=epochs, 
    )

    save_lstm_model(tokenizer, model, model_name)

def continue_train(old_model_name="test", new_model_name="test", epochs=10):
    text = get_data()

    # *****
    # load
    tokenizer, model = load_lstm_model(old_model_name)

    # max_len_sequence = model.input.shape[1]
    vocab_size = len(tokenizer.word_index)+1
    
    # *****
    #
    X, y, _ = text_encoder(text, vocab_size, tokenizer)
    
    model.fit(
        X, 
        y, 
        batch_size=128, 
        epochs=epochs, 
    )

    save_lstm_model(tokenizer, model, new_model_name)

# *************************************************
# generate seed and predict / generate text
# see generating method: https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
# *************************************************
def generate(model_name="test"):
    tokenizer, model = load_lstm_model(model_name)
    max_len_sequence = model.input.shape[1]
    vocab_size = len(tokenizer.word_index)+1
    
    seed_text = "vorsichtig in eine schüssel geben"

    for _ in range(1000):
        x = tokenizer.texts_to_sequences([seed_text])[0]
        x = pad_sequences([x], maxlen=max_len_sequence)
        
        prediction = model.predict_classes(x, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == prediction:
                output_word = word
                break
        if output_word == "eos":
            break
        seed_text += f" {output_word}"
    print(seed_text)

# train("dessert_recipes_2/test_300", 300)
# continue_train("dessert_recipes_2/test_300", "dessert_recipes_2/test_500", 200)
generate("dessert_recipes_2/test_300")
