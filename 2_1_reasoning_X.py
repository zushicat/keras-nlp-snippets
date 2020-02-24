# https://appliedmachinelearning.blog/2019/05/02/building-end-to-end-memory-network-for-question-answering-system-on-babi-facebook-data-set-python-keras-part-2/
# https://keras.io/examples/babi_memnn/
# (https://github.com/jalajthanaki/Chatbot_based_on_bAbI_dataset_using_Keras/blob/master/main.py)
#
#
# simpler rnn implementation:
# https://keras.io/examples/babi_rnn/
# https://smerity.com/articles/2015/keras_qa.html

from __future__ import print_function

import json

import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, concatenate, dot
from keras.layers import LSTM, GRU
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from functools import reduce
# import tarfile
import numpy as np
import re

train_model = 0
train_epochs = 100
load_model = 1
batch_size = 32
lstm_size = 64
test_qualitative = 1
user_questions = 0



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.findall('[\w]+|[.,!?;]', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    num_stories = 0
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            num_stories += 1
            if num_stories == 2:
              break
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

# **************************************************
#
# **************************************************
challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact': '../../data/bAbI/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts': '../../data/bAbI/tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact'
challenge = challenges[challenge_type]

train_stories_path = challenge.format('train')
with open(train_stories_path) as f:
  train_stories = get_stories(f, only_supporting=True)
# test_stories_path = challenge.format('test')
# with open(test_stories_path) as f:
#   test_stories = get_stories(f, only_supporting=True)


# **************************************************
#
# **************************************************
vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1  # Reserve 0 for masking via pad_sequences
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
idx_word = dict((i+1, c) for i,c in enumerate(vocab))

inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
# inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)


# **************************************************
# This model is a little more complex
# see: https://appliedmachinelearning.blog/2019/05/02/building-end-to-end-memory-network-for-question-answering-system-on-babi-facebook-data-set-python-keras-part-2/
# and compare with model diagram
# **************************************************
def build_model(story_maxlen, query_maxlen, vocab_size, hidden_layer):
    story_inputs = Input((story_maxlen,), name="story_inputs")
    question_inputs = Input((query_maxlen,), name="question_inputs")

    # *********************************
    # input memory representation
    input_encoder_m = Sequential(name="input_memory_representation")
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=hidden_layer))
    input_encoder_m.add(Dropout(0.3))

    # *********************************
    # output memory representation
    input_encoder_c = Sequential(name="output_memory_representation")
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
    input_encoder_c.add(Dropout(0.3))

    # *********************************
    # question memory representation
    question_encoder = Sequential(name="question_memory_representation")
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
    question_encoder.add(Dropout(0.3))

    # *********************************
    # encode input sequence and questions (originally indices) to sequences of dense vectors
    embedding_input_m = input_encoder_m(story_inputs)
    embedding_input_c = input_encoder_c(story_inputs)
    embedding_question = question_encoder(question_inputs)

    # *********************************
    # probability vector
    # compute a 'match' between first input vector sequence and question vector sequence
    probability_vector = dot([embedding_input_m, embedding_question], axes=(2, 2), name="probability_vector")
    probability_vector = Activation('softmax')(probability_vector)

    # *********************************
    # response / output vector
    # add the match matrix with second input vector sequence
    output_vector = add([probability_vector, embedding_input_c], name="output_vector_o") 
    output_vector = Permute((2, 1))(output_vector)  #  i.e. (None, 7, 4) -> (None, 4, 7)

    # *********************************
    # result vector -> 'answer'
    # concatenate response vector with question vector sequence
    result_vector = concatenate([output_vector, embedding_question], name="result_vector")

    decoder_lstm = LSTM(hidden_layer, dropout=0.3, name="answer_lstm")
    decoder_outputs = decoder_lstm(result_vector)
    decoder_outputs = Dense(vocab_size, activation='softmax', name="dense")(decoder_outputs)

    # *********************************
    #
    model = Model([story_inputs, question_inputs], decoder_outputs)
    model.summary()

    return model


def train():
    hidden_layer = 64
    model = build_model(story_maxlen, query_maxlen, vocab_size, hidden_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([inputs_train, queries_train], answers_train, batch_size, train_epochs)


# # **************************************************
# #
# # **************************************************
# if load_model == 1:
#     model = keras.models.load_model('models/2_1_reasoning/model.h5')

# if train_model == 1:
#     # train, batch_size = 32 and epochs = 120
#     model.fit([inputs_train, queries_train], answers_train, batch_size, train_epochs,
#           validation_data=([inputs_test, queries_test], answers_test))
#     model.save('models/2_1_reasoning/model.h5')


# # **************************************************
# #
# # **************************************************
# if test_qualitative == 1:
#     print('-------------------------------------------------------------------------------------------')
#     print('Qualitative Test Result Analysis')
#     for i in range(0,1):
#         current_inp = test_stories[i]
#         print(current_inp)
#         print("----")
#         current_story, current_query, current_answer = vectorize_stories([current_inp], word_idx, story_maxlen, query_maxlen)
#         current_prediction = model.predict([current_story, current_query])
#         current_prediction = idx_word[np.argmax(current_prediction)]
#         print(' '.join(current_inp[0]), ' '.join(current_inp[1]), '| Prediction:', current_prediction, '| Ground Truth:', current_inp[2])

# # **************************************************
# #
# # **************************************************
# if user_questions == 1:
#     print('-------------------------------------------------------------------------------------------')
#     print('Custom User Queries (Make sure there are spaces before each word)')
#     while 1:
#         print('-------------------------------------------------------------------------------------------')
#         print('Please input a story')
#         user_story_inp = raw_input().split(' ')
#         print('Please input a query')
#         user_query_inp = raw_input().split(' ')
#         print (user_story_inp, user_query_inp)
#         user_story, user_query, user_ans = vectorize_stories([[user_story_inp, user_query_inp, '.']], word_idx, story_maxlen, query_maxlen)
#         user_prediction = model.predict([user_story, user_query])
#         user_prediction = idx_word[np.argmax(user_prediction)]
#         print('Result')
#         print(' '.join(user_story_inp), ' '.join(user_query_inp), '| Prediction:', user_prediction)

