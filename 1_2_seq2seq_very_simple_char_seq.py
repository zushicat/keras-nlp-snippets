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
# generate dataset
# ***************************************************
def get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, num_sequences):
    """
    Generate corresponding question and answer lists
    - vocab: letters from alphabet
    - answers: reversed subset of questions i.e. q: [a,b,b,a,b], a: [a,b,b] -> [b,b,a]
    """
    # generate list of lowercase letters of alphabet
    letters = [chr(i) for i in range(ord('a'),ord('z')+1)]

    # generate a sequence of random integers between 1 and i.e. (2+1)-1 -> (1, 2)
    def generate_sequence(max_len_sequence_in, vocab_size):
	    return [randint(1, vocab_size-1) for _ in range(max_len_sequence_in)]

    # create list of chars
    questions, answers = list(), list()
    for _ in range(num_sequences):
        # generate source sequence (shift i left for letter array: 1 -> 0 etc.)
        question = [letters[i-1] for i in generate_sequence(max_len_sequence_in, vocab_size)]
        questions.append(question)

        # create target sequence
        answer = question[:max_len_sequence_out]
        answer.reverse()
        answers.append(answer)
        
    return questions, answers

# ***************************************************
# encoder / decoder
# ***************************************************
def encoder(questions, answers, max_len_sequence_out, vocab_size, tokenizer):
    # create input target sequence
    answers_in = [answer[:-1] for answer in answers]
    
    """
    now i.e.
    questions:  [['a', 'a', 'b']]
    answers:    [['a', 'a']]
    answers_in: [['a']]
    """

    question_sequences = tokenizer.texts_to_sequences(questions)
    answers_sequences = tokenizer.texts_to_sequences(answers)
    answers_in_sequences = tokenizer.texts_to_sequences(answers_in)
    """
    now i.e.
    questions:  [[1, 1, 2]]
    answers:    [[1, 1]]
    answers_in: [[1]]
    """
    
    padded_question_sequences = pad_sequences(question_sequences)
    padded_answers_sequences = pad_sequences(answers_sequences)
    padded_answers_in_sequences = pad_sequences(answers_in_sequences, maxlen=max_len_sequence_out)
    """
    now i.e.
    questions:  [[1 1 2]]
    answers:    [[1 1]]
    answers_in: [[0 1]]
    """
    
    # encode
    questions_encoded = to_categorical(padded_question_sequences, num_classes=vocab_size)
    answers_in_encoded = to_categorical(padded_answers_in_sequences, num_classes=vocab_size)
    answers_encoded = to_categorical(padded_answers_sequences, num_classes=vocab_size)
    """
    now questions i.e.
    [
        [
            [0. 1. 0.]
            [0. 1. 0.]
            [0. 0. 1.]
        ]
    ]
    """
    
    return questions_encoded, answers_in_encoded, answers_encoded  # X1, X2, y

def decoder(encoded_sequences, tokenizer):
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
# returns model, inference_encoder and inference_decoder
# ***************************************************
def define_models(vocab_size, n_units):
    '''
    n_units: dimensionality of the output space in the encoder and decoder models, e.g. 128 or 256.
    '''
    # ****************
    # model
    # ****************
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
    
    # ***
    # create
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # ****************
    # inference ENcoder
    # ****************
    # create
    encoder_model = Model(encoder_inputs, encoder_states)

    # ****************
    # inference DEcoder
    # ****************
    # define
    # ***
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # ***
    # create
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    

    return model, encoder_model, decoder_model


# ***************************************************
# predicct/generate 'answer' to given 'question' sequence
# ***************************************************
def predict_sequence(inference_encoder, inference_decoder, question_sequence, max_len_sequence_out, vocab_size):
    '''
    infenc: Encoder model used when making a prediction for a new source sequence.
    infdec: Decoder model use when making a prediction for a new source sequence.
    source:Encoded source sequence.
    '''
    # ***
    # encode
    state = inference_encoder.predict(question_sequence)
    
    # ***
    # start of sequence input
    answer_sequence = np.array([0.0 for _ in range(vocab_size)]).reshape(1, 1, vocab_size)
    
    # ***
    # collect predictions
    output = list()
    for t in range(max_len_sequence_out):
        # predict next char
        yhat, h, c = inference_decoder.predict([answer_sequence] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        answer_sequence = yhat
    
    return np.array(output)



# ***********************************************************************************
# 
# ***********************************************************************************
vocab_size = 2 + 1  # num features (generated vocabularyof letters in alphabet 0 -> max len alphabet)
max_len_sequence_in = 4  # longest 'question' (is len of all generated source sequences)
max_len_sequence_out = 3 # longest 'answer' (is len of all generated target sequences)
num_trainsets = 3000  # num generated trainsets (should certainly depend on vocab_size and max len sequences in/out)

# ***************************************************
# generate training dataset
# ***************************************************
questions, answers = get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, num_trainsets)

# fit tokenize on all text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# encode questions and answers
X1, X2, y = encoder(questions, answers, max_len_sequence_out, vocab_size, tokenizer)

# ***************************************************
# define model
# ***************************************************
model, inference_encoder, inference_decoder = define_models(vocab_size, 128)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# ***************************************************
# train model
# ***************************************************
model.fit([X1, X2], y, epochs=2)

# ***************************************************
# predict some examples and count correct predictions (evaluate LSTM)
# ***************************************************
total, correct = 10, 0
for i in range(total):
    """
    Generated questions and answers initially have this form: [['a', 'b', 'a', 'b']],
    so in order to get whole string 'a b a b' just use decoder instead of: ' '.join(question_test[0])

    See: comments in encoder / decoder
    """
    # create testset with 1 question/answer
    questions_test, answers_test = get_dataset(max_len_sequence_in, max_len_sequence_out, vocab_size, 1)
    # encode
    X_test, _, y_test = encoder(questions_test, answers_test, max_len_sequence_out, vocab_size, tokenizer)
    
    # predict y (answer) and decode to text
    # max_len_sequence_out -> stop condition
    predicted_sequence = predict_sequence(inference_encoder, inference_decoder, X_test, max_len_sequence_out, vocab_size)
    predicted_text = decoder([predicted_sequence], tokenizer)[0]

    # decode back to 1 text line
    question_text = decoder(X_test, tokenizer)[0]
    answer_text = decoder(y_test, tokenizer)[0]
    
    # print(f"{i} --> X: {question_text} | y: {answer_text} | yhat: {predicted_text} ---> {answer_text == predicted_text}")

    if predicted_text == answer_text:
        correct += 1

print(f"Accuracy: {float(correct)/float(total)*100.0}")
