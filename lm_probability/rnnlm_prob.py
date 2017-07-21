
# coding: utf-8

# In[1]:

import numpy, csv, pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.models import load_model


# In[2]:

with open('../tokenizer_96000.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

rnn = load_model('../generation_rnn_96000.h5')


# In[24]:

def get_prob(seq):
    probs = []
    seq = tokenizer.texts_to_sequences([seq])[0]
    seq = numpy.array(seq)[None] #reshape as (1, seq_length)
    for idx in range(seq.shape[-1] - 1):
        prob = rnn.predict_on_batch(seq[:, idx])[0,-1][seq[0, idx +1]] #get prob of next word
        probs.append(prob)
    return numpy.mean(probs) #return average probability of words in sequence


# In[27]:

get_prob("Jane was working at a diner. Suddenly, a customer barged up to the counter.")


# In[26]:

get_prob("Jane was working at a diner. Dave swam away from the shark.")

