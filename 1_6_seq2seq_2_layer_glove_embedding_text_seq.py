# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
# https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639
import json
import pickle

from _load_data import load_data_from_yml, tag_answers

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, model_from_json, save_model, load_model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant

# ***************************************************
# get source and target (questions and answers)
# ***************************************************
def get_data():
    questions, answers = load_data_from_yml("../../data/chatbot_data/karin", "conversations")
    answers = tag_answers(answers)  # add <BOS> and <EOS> at start/end of answer text line

    return questions, answers


# ***************************************************
# create word embedding matrix
# https://keras.io/examples/pretrained_word_embeddings/
# ***************************************************
def word_embedding_glove(tokenizer, vocab_size):
    EMBEDDING_DIM = 100

    embeddings_index = {}
    with open("../../data/glove/glove.6B.100d.txt") as f:
        lines = f.read().split("\n")
        for line in lines:
            try:
                line = line.split()
                word = line[:1]
                if tokenizer.word_index.get(word) is None:  # pre-filter for speed
                    continue
                coefs = np.array([float(val) for val in line[1:]])
                embeddings_index[word] = coefs
            except:
                continue
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
        

# ***************************************************
# encoder / decoder
# <BOS>/<EOS> tagging: https://stackoverflow.com/a/55146904
# ! for use with embedding in model: only decoder_out is one-hot encoded !
# ***************************************************
def text_encoder(questions, answers, max_len_sequence, vocab_size, tokenizer):
    '''
    Text encoding for embedding layer: index based with max len
    answers: <BOS> abc <EOS> -> answers_in: <BOS> abc | answers_out: abc <EOS>
    '''
    answers_in = [answer for answer in answers]  # decoder input
    answers_in = [" ".join(x.split()[:-1]) for x in answers_in]
    answers_out = [" ".join(x.split()[1:]) for x in answers]  # decoder

    # sequences
    question_sequences = tokenizer.texts_to_sequences(questions)
    answers_out_sequences = tokenizer.texts_to_sequences(answers_out)
    answers_in_sequences = tokenizer.texts_to_sequences(answers_in)
    
    # padding
    padded_question_sequences = pad_sequences(question_sequences, maxlen=max_len_sequence, padding="post")
    padded_answers_out_sequences = pad_sequences(answers_out_sequences, maxlen=max_len_sequence, padding="post")
    padded_answers_in_sequences = pad_sequences(answers_in_sequences, maxlen=max_len_sequence, padding="post")
    
    encoded_answers_out_sequences = to_categorical(padded_answers_out_sequences, num_classes=vocab_size)

    return padded_question_sequences, padded_answers_in_sequences, encoded_answers_out_sequences

def text_decoder(encoded_sequences, tokenizer):
    """
    i.e.
    --> a sequence (in sequences)
    [
        [0. 1. 0.]
        [0. 1. 0.]
        [0. 0. 1.]
    ] 
    --> [[1, 1, 2]] after one hot encoding 
    --> ['b b a']  after sequences_to_texts
    """
    one_hot_decoder = list()
    for encoded_sequence in encoded_sequences:  # each 'sentence'
        one_hot_decoder.append([np.argmax(vector) for vector in encoded_sequence])  # append list of indices to list
    decoded_sentences = tokenizer.sequences_to_texts(one_hot_decoder)
    
    return decoded_sentences


# ***************************************************
# buld model and inference encoder / decoder from model
# https://stackoverflow.com/a/56448284
# is a little different from model without embedding layer
# see: https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7
# with 2 layer (model / inf. enc/dec): 
# https://stackoverflow.com/questions/50915634/multilayer-seq2seq-model-with-lstm-in-keras
# ***************************************************
def build_model(vocab_size, hidden_layer, max_len_sequence, embedding_matrix):
    # ***
    # encoder
    # ***
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding_layer = Embedding(
        input_dim = vocab_size, 
        output_dim = 100,
        name = "encoder_embedding",
        input_length = max_len_sequence,
        embeddings_initializer=Constant(embedding_matrix),
        trainable = True,
        mask_zero="True"
    )  # encoder_embedding_layer = Embedding(vocab_size, hidden_layer, input_length=max_len_sequence, mask_zero="True", name="encoder_embedding")
    encoder_embedding = encoder_embedding_layer(encoder_inputs)
    
    encoder_outputs, state_h_1, state_c_1 = LSTM(hidden_layer, return_sequences=True, return_state=True, dropout=0.5, name="enc_lstm_1")(encoder_embedding)
    _, state_h_2, state_c_2 = LSTM(hidden_layer, return_state=True, name="enc_lstm_2")(encoder_outputs)
    encoder_states_1 = [state_h_1, state_c_1]
    encoder_states_2 = [state_h_2, state_c_2]

    # ***
    # decoder
    # ***
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding_layer = Embedding(
        input_dim = vocab_size, 
        output_dim = 100,
        name = "decoder_embedding",
        input_length = max_len_sequence,
        embeddings_initializer=Constant(embedding_matrix),
        trainable = True,
        mask_zero="True"
    )  # Embedding(vocab_size, hidden_layer, input_length=max_len_sequence, mask_zero="True", name="decoder_embedding")
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    
    decoder_lstm_1 = LSTM(hidden_layer, return_sequences=True, return_state=True, dropout=0.5, name="dec_lstm_1")
    decoder_outputs_1, _, _ = decoder_lstm_1(decoder_embedding, initial_state=encoder_states_1)

    decoder_lstm_2 = LSTM(hidden_layer, return_sequences=True, return_state=True, name="dec_lstm_2")
    decoder_outputs_2, _, _ = decoder_lstm_2(decoder_outputs_1, initial_state=encoder_states_2)
    
    decoder_dense = Dense(vocab_size, activation='softmax', name="dense")
    decoder_outputs = decoder_dense(decoder_outputs_2)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    return model

def build_inference_encoder_decoder(model, hidden_layer):
    # ***
    # encoder
    # ***
    encoder_inputs = model.input[0]   # input_1
    
    _, state_h_1, state_c_1 = model.get_layer("enc_lstm_1").output   # lstm_1
    _, state_h_2, state_c_2 = model.get_layer("enc_lstm_2").output   # lstm_2
    
    encoder_states = [state_h_1, state_c_1, state_h_2, state_c_2]
    
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # ***
    # decoder
    # ***
    decoder_state_input_h_1 = Input(shape=(hidden_layer,))
    decoder_state_input_c_1 = Input(shape=(hidden_layer,))
    decoder_state_input_h_2 = Input(shape=(hidden_layer,))
    decoder_state_input_c_2 = Input(shape=(hidden_layer,))
    
    decoder_states_inputs = [
        decoder_state_input_h_1, 
        decoder_state_input_c_1,
        decoder_state_input_h_2, 
        decoder_state_input_c_2
    ]
    
    decoder_inputs = model.input[1]
    decoder_embedding = model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm_1 = model.get_layer("dec_lstm_1")
    decoder_lstm_2 = model.get_layer("dec_lstm_2")
    decoder_dense = model.get_layer("dense")

    decoder_outputs_1, state_h_1, state_c_1 = decoder_lstm_1(decoder_embedding, initial_state=decoder_states_inputs[:2])
    decoder_outputs_2, state_h_2, state_c_2 = decoder_lstm_2(decoder_outputs_1, initial_state=decoder_states_inputs[-2:])

    decoder_states = [state_h_1, state_c_1, state_h_2, state_c_2]
    decoder_outputs = decoder_dense(decoder_outputs_2)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model


# ***************************************************
# save / load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# ***************************************************
def save_lstm_model(tokenizer, model, model_name, max_len_sequence_out):
    model_config = model.to_json(indent=2)  # serialize model to JSON (str)
    
    # add additinal info: length of longest answer text
    model_config = json.loads(model_config)
    model_config["max_len_sequence_out"] = max_len_sequence_out 
    model_config = json.dumps(model_config, indent=2)
    
    with open(f"models/1_6_seq2seq_chatbot_seq/{model_name}_config.json", "w") as f:
        f.write(model_config)
    model.save_weights(f"models/1_6_seq2seq_chatbot_seq/{model_name}_weights.h5")  # serialize weights to HDF5

    with open(f"models/1_6_seq2seq_chatbot_seq/{model_name}_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_lstm_model(model_name):
    with open(f"models/1_6_seq2seq_chatbot_seq/{model_name}_config.json") as f:
        model_config = f.read()
    model = model_from_json(model_config)
    model.load_weights(f"models/1_6_seq2seq_chatbot_seq/{model_name}_weights.h5")

    with open(f"models/1_6_seq2seq_chatbot_seq/{model_name}_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    
    n_units = None
    config_data = json.loads(model_config)
    max_len_sequence_out = config_data["max_len_sequence_out"]
    for layer in config_data["config"]["layers"]:
        if layer["name"] == "lstm_1":
            n_units = layer["config"]["units"]
       
    return tokenizer, model, n_units, max_len_sequence_out


# ***********************************************************************************
# 
# ***********************************************************************************
def train(model_name="test", epochs=10):
    # **********
    # generate training dataset
    # **********
    questions, answers = get_data()
    questions = questions[:20]
    answers = answers[:20]

    # fit tokenize on all text
    tokenizer = Tokenizer(oov_token="<UNKNOWN>")
    tokenizer.fit_on_texts(questions + answers)

    vocab_size = len(tokenizer.word_index)+1
    max_len_sequence_in = max([len(x.split()) for x in questions])
    max_len_sequence_out = max([len(x.split()) for x in answers])

    max_len_sequence = max(max_len_sequence_in, max_len_sequence_out)
    
    # **********
    # create embedding matrix with glove word embeddings
    # **********
    embedding_matrix = word_embedding_glove(tokenizer, vocab_size)

    # **********
    # encode questions and answers
    # when embeddings are used: X1, X2 NOT one-hot-encoded, padded only / y is one-hot-encoded
    # **********
    X1, X2, y = text_encoder(questions, answers, max_len_sequence, vocab_size, tokenizer)
    
    # **********
    # build & train model
    # **********
    # there are some rules of thumb on how to determine number of hidden layer (latent_dim)
    # it should be at least 1.5 times of Ni
    # **
    hidden_layer = (2 * vocab_size) + 1  # Nh = (2 * Ni) + 1 (Ni and No are equal in this model) 
    
    # build model and inference encoder/decoder models
    model = build_model(vocab_size, hidden_layer, max_len_sequence, embedding_matrix)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([X1, X2], y, epochs=epochs, batch_size=32)

    # **********
    # save
    # add additional info to config json about max length answer sequence
    # **********
    save_lstm_model(tokenizer, model, model_name, max_len_sequence_out)


# ***************************************************
# predict/generate 'answer' to given 'question' sequence
# 2D shape from embedding: https://towardsdatascience.com/implementing-neural-machine-translation-using-keras-8312e4844eb8
# (see also https://keras.io/examples/lstm_seq2seq_restore/ -> decode_sequence(input_seq) )
# prediction with 2 lstm layer: https://stackoverflow.com/questions/50915634/multilayer-seq2seq-model-with-lstm-in-keras
# ***************************************************
def predict_sequence(
        inference_encoder, 
        inference_decoder, 
        question_sequence, 
        max_len_sequence_out, 
        tokenizer
    ):
    # ***
    # encode
    state = inference_encoder.predict(question_sequence)
    
    # ***
    # initialize empty answer sequesence sequence
    # ***
    decoder_input = np.zeros((1,1))
    decoder_input[0, 0] = tokenizer.word_index.get("bos")
    
    # ***
    # collect predictions
    decoded_sequence = list()
    for _ in range(max_len_sequence_out):
        decoder_output, h_1, c_1, h_2, c_2 = inference_decoder.predict([decoder_input] + state)
        state = [h_1, c_1, h_2, c_2]  # update state (4 states because 2 lstm layer)

        # get token text of best prediction
        # check break condition: stopword from trained answers <END> --> end
        sampled_token_index = np.argmax(decoder_output[0, -1, :])
        sample_token = tokenizer.sequences_to_texts([[sampled_token_index]])[0]
        if sample_token == "eos":
            break
        decoded_sequence.append(sample_token)

        decoder_input = np.zeros((1,1))
        decoder_input[0, 0] = sampled_token_index
        
    return " ".join(decoded_sequence)


# ***************************************************
# predict some examples and count correct predictions (evaluate LSTM)
# https://keras.io/examples/lstm_seq2seq_restore/
# ***************************************************
def predict(model_name):
    # ***********
    # load data
    # ***********
    tokenizer, model, n_units, max_len_sequence_out = load_lstm_model(model_name)
    model.summary()
    vocab_size = len(tokenizer.word_index)+1
    inference_encoder, inference_decoder = build_inference_encoder_decoder(model, n_units)
    
    # ***********
    # predict sequence
    # ***********
    questions_test = ["do you like dogs?"]
    answers_test = ["yes"]

    for i, question in enumerate(questions_test):
        question = [question]
        answer = [answers_test[i]]

        # encode text and get prediction
        X_test, _, y_test = text_encoder(question, answer, max_len_sequence_out, vocab_size, tokenizer)
        prediction = predict_sequence(inference_encoder, inference_decoder, X_test, max_len_sequence_out, tokenizer)
        
        print(f"X: {question[0]} | y: {answer[0]} | yhat: {prediction} ---> {answer[0] == prediction}")
        print("---")


train(model_name="conversation", epochs=300)
predict(model_name="conversation")
