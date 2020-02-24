# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
import json

from random import randint
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# ***************************************************
# generate encoded sequences of numbers
# ***************************************************
def get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, num_sequences):
    # generate a sequence of random integers
    def generate_sequence(max_len_sequence_in, vocab_size):
	    return [randint(1, vocab_size-1) for _ in range(max_len_sequence_in)]

    X1, X2, y = list(), list(), list()
    for _ in range(num_sequences):
        # generate source sequence
        source = generate_sequence(max_len_sequence_in, vocab_size)
        
        # define padded target sequence
        target = source[:max_len_sequence_out]
        target.reverse()

        # create padded input target sequence
        target_in = [0] + target[:-1]

        # print(source)
        # print(target_in)
        # print(target)
        # print("----")

        # encode
        source_encoded = to_categorical([source], num_classes=vocab_size)
        target_encoded = to_categorical([target], num_classes=vocab_size)
        target_in_encoded = to_categorical([target_in], num_classes=vocab_size)
        
        # print(source_encoded)
        # print("----")

        # append encoded sequence lists to lists
        X1.append(source_encoded)
        X2.append(target_in_encoded)
        y.append(target_encoded)
    
    # remove 1 axis (otherwise 4D shape instead of 3D)
    X1 = np.squeeze(np.array(X1), axis=1) 
    X2 = np.squeeze(np.array(X2), axis=1) 
    y = np.squeeze(np.array(y), axis=1) 

    return X1, X2, y

# ***************************************************
# returns train, inference_encoder and inference_decoder models
# ***************************************************
def define_models(vocab_size, n_units):
    '''
    n_units: dimensionality of the output space in the encoder and decoder models, e.g. 128 or 256.
    '''
    # ***
    # define training encoder
    encoder_inputs = Input(shape=(None, vocab_size))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # ***
    # define training decoder
    decoder_inputs = Input(shape=(None, vocab_size))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # ***
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # ***
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # ***
    # return all models
    return model, encoder_model, decoder_model

# ***************************************************
# generate target given source sequence
# ***************************************************
def predict_sequence(infenc, infdec, source, max_len_sequence_out, vocab_size):
    '''
    infenc: Encoder model used when making a prediction for a new source sequence.
    infdec: Decoder model use when making a prediction for a new source sequence.
    source:Encoded source sequence.
    '''
    # ***
    # encode
    state = infenc.predict(source)
    
    # ***
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(vocab_size)]).reshape(1, 1, vocab_size)
    
    # ***
    # collect predictions
    output = list()
    for t in range(max_len_sequence_out):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    
    return np.array(output)

# ***************************************************
# decode a one hot encoded string
# returns index of 1. in each vector i.e. [[0. 0. 1.]] -> [0. 0. 1.] -> 2
# ***************************************************
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

# ***************************************************
# configure problem
# ***************************************************
vocab_size = 4 + 1  # num features (generated vocabulary: numbers 1->n)
max_len_sequence_in = 6  # longest 'question' (is len of all generated source sequences)
max_len_sequence_out = 3 # longest 'answer' (is len of all generated target sequences)
num_trainsets = 5000  # num generated trainsets (certainly depends on vocab_size and max len sequences in/out)

# ***************************************************
# generate training dataset
# ***************************************************
X1, X2, y = get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, num_trainsets)

# ***************************************************
# define model
# ***************************************************
train, infenc, infdec = define_models(vocab_size, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ***************************************************
# train model
# ***************************************************
train.fit([X1, X2], y, epochs=10)

# ***************************************************
# evaluate LSTM
# ***************************************************
total, correct = 100, 0
for _ in range(total):
    X_test, _, y_test = get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, 1)
    prediction = predict_sequence(infenc, infdec, X_test, max_len_sequence_out, vocab_size)
    if np.array_equal(one_hot_decode(y_test[0]), one_hot_decode(prediction)):
        correct += 1
print(f'Accuracy: {float(correct)/float(total)*100.0}')

# ***************************************************
# check some examples
# ***************************************************
for _ in range(10):
    X_test, _, y_test = get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, 1)
    prediction = predict_sequence(infenc, infdec, X_test, max_len_sequence_out, vocab_size)
    print(f'X={one_hot_decode(X1[0])} y={one_hot_decode(y_test[0])}, yhat={one_hot_decode(prediction)}')