# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
# https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639
import json
import pickle

from _load_data import load_data_from_yml, tag_answers

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, save_model, load_model

# ***************************************************
# get source and target (questions and answers)
# ***************************************************
def get_data():
    questions, answers = load_data_from_yml("../../data/chatbot_data/shubham0204/chatbot_nlp", "conversations")
    answers = tag_answers(answers)  # add <BOS> and <EOS> at start/end of answer text line

    return questions, answers


# ***************************************************
# encoder / decoder
# <BOS>/<EOS> tagging: https://stackoverflow.com/a/55146904
# ***************************************************
def text_encoder(questions, answers, max_len_sequence_out, vocab_size, tokenizer):
    '''
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
    padded_question_sequences = pad_sequences(question_sequences)
    padded_answers_out_sequences = pad_sequences(answers_out_sequences, maxlen=max_len_sequence_out, padding="post")
    padded_answers_in_sequences = pad_sequences(answers_in_sequences, maxlen=max_len_sequence_out, padding="post")
    
    # encoding
    questions_encoded = to_categorical(padded_question_sequences, num_classes=vocab_size)
    answers_in_encoded = to_categorical(padded_answers_in_sequences, num_classes=vocab_size)
    answers_out_encoded = to_categorical(padded_answers_out_sequences, num_classes=vocab_size)
    
    return questions_encoded, answers_in_encoded, answers_out_encoded  # X1, X2, y

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
# ***************************************************
def build_model(vocab_size, hidden_layer):
    # ****************
    # model
    # ****************
    # define training encoder
    encoder_inputs = Input(shape=(None, vocab_size))

    encoder = LSTM(hidden_layer, return_state=True, return_sequences=True)
    
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # ***
    # define training decoder
    decoder_inputs = Input(shape=(None, vocab_size))

    decoder = LSTM(hidden_layer, return_sequences=True, return_state=True)
    
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    return model

def build_inference_encoder_decoder(model, hidden_layer):
    # ***
    # encoder
    # ***
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    
    encoder_model = Model(encoder_inputs, encoder_states)

    # ***
    # decoder
    # ***
    decoder_inputs = model.input[1]   # input_2
    
    decoder_state_input_h = Input(shape=(hidden_layer,), name='input_3')
    decoder_state_input_c = Input(shape=(hidden_layer,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder = model.layers[3]
    
    decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    
    decoder_states = [state_h, state_c]
    
    decoder_dense = model.layers[4]  # dense
    decoder_outputs = decoder_dense(decoder_outputs)
    
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
    
    with open(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_config.json", "w") as f:
        f.write(model_config)
    model.save_weights(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_weights.h5")  # serialize weights to HDF5

    with open(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_lstm_model(model_name):
    with open(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_config.json") as f:
        model_config = f.read()
    model = model_from_json(model_config)
    model.load_weights(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_weights.h5")

    with open(f"models/chatbot_seq2seq_simple_text_seq/{model_name}_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    
    n_units = None
    config_data = json.loads(model_config)
    max_len_sequence_out = config_data["max_len_sequence_out"]
    for layer in config_data["config"]["layers"]:
        if layer["name"] == "lstm_1":
            n_units = layer["config"]["units"]
       
    return tokenizer, model, n_units, max_len_sequence_out

# ***************************************************
# predict/generate 'answer' to given 'question' sequence
# see also https://keras.io/examples/lstm_seq2seq_restore/ -> decode_sequence(input_seq)
# ***************************************************
def predict_sequence(
        inference_encoder, 
        inference_decoder, 
        question_sequence, 
        max_len_sequence_out, 
        vocab_size,
        tokenizer
    ):
    # ***
    # encode
    state = inference_encoder.predict(question_sequence)
    
    # ***
    # initialize empty sequence
    answer_sequence = np.array([0.0 for _ in range(vocab_size)]).reshape(1, 1, vocab_size)
    
    # ***
    # collect predictions
    output = list()
    for _ in range(max_len_sequence_out):
        # predict next char
        output_tokens, h, c = inference_decoder.predict([answer_sequence] + state)
        
        # get token text of best prediction
        # check break condition: stopword from trained answers <END> --> end
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sample_token = tokenizer.sequences_to_texts([[sampled_token_index]])[0]
        if sample_token == "eos":
            break
        
        # store prediction
        output.append(output_tokens[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        answer_sequence = output_tokens
    
    return np.array(output)



# ***********************************************************************************
# 
# ***********************************************************************************
def train(model_name="test", epochs=10):
    # **********
    # generate training dataset
    # **********
    questions, answers = get_data()
    # questions = questions[:50]
    # answers = answers[:50]

    # fit tokenize on all text
    tokenizer = Tokenizer(oov_token="<UNKNOWN>")
    tokenizer.fit_on_texts(questions + answers)

    vocab_size = len(tokenizer.word_index)+1
    max_len_sequence_out = max([len(x.split()) for x in answers])

    # encode questions and answers
    X1, X2, y = text_encoder(questions, answers, max_len_sequence_out, vocab_size, tokenizer)

    # **********
    # build model
    # **********
    # there are some rules of thumb on how to determine number of hidden layer
    # it should be at least 1.5 times of Ni
    # **
    hidden_layer = (2 * vocab_size) + 1  # Nh = (2 * Ni) + 1 (Ni and No are equal in this model) 
    
    # build model and inference encoder/decoder models
    model = build_model(vocab_size, hidden_layer)
    inference_encoder, inference_decoder = build_inference_encoder_decoder(model, hidden_layer)
    
    # rmsprop seems to work much better than adam (both with default leraning rates) for this setup
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # **********
    # train model
    # **********
    model.fit([X1, X2], y, epochs=epochs)

    # **********
    # save
    # add additional info to config json about max length answer sequence
    # **********
    save_lstm_model(tokenizer, model, model_name, max_len_sequence_out)


# ***************************************************
# predict some examples and count correct predictions (evaluate LSTM)
# https://keras.io/examples/lstm_seq2seq_restore/
# ***************************************************
def predict(model_name):
    # ***********
    # load data
    # ***********
    tokenizer, model, n_units, max_len_sequence_out = load_lstm_model(model_name)
    vocab_size = len(tokenizer.word_index)+1
    inference_encoder, inference_decoder = build_inference_encoder_decoder(model, n_units)
    model.summary()
    # ***********
    # predict sequence
    # ***********
    questions_test = ["What is AI?"]
    answers_test = ["artificial intelligence is the branch of engineering and science devoted to constructing machines that think"]

    for i, question in enumerate(questions_test):
        question = [question]
        answer = [answers_test[i]]

        # encode text
        X_test, _, y_test = text_encoder(question, answer, max_len_sequence_out, vocab_size, tokenizer)
        
        # predict answer and decode to text
        predicted_sequence = predict_sequence(inference_encoder, inference_decoder, X_test, max_len_sequence_out, vocab_size, tokenizer)
        prediction = text_decoder([predicted_sequence], tokenizer)[0]
        
        print(f"X: {question[0]} | y: {answer[0]} | yhat: {prediction} ---> {answer[0] == prediction}")
        print("---")


# train(model_name="test", epochs=1)
predict(model_name="test")
