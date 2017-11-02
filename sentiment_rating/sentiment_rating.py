
# coding: utf-8

# # <font color='#6629b2'>Predicting sentiment ratings with recurrent neural networks using Keras</font>
# ### https://github.com/roemmele/keras-rnn-demo/sentiment-rating
# by Melissa Roemmele, 10/23/17, roemmele @ ict.usc.edu

# ## <font color='#6629b2'>Overview</font>
# 
# I am going to show how to use the Keras library to build a recurrent neural network (RNN) model that predicts sentiment ratings for text sequences. Specifically, the model will predict the ratings associated with movie reviews.
# 
# ### <font color='#6629b2'>Recurrent Neural Networks</font>
# 
# RNNs are a general framework for modeling sequence data and are particularly useful for natural language processing tasks. At a high level, RNN encode sequences via a set of parameters (weights) that are optimized to predict some output variable. The intention of this tutorial is to demonstrate the code needed to assemble an RNN model using the Keras library, as well as some data processing tools that facilitate building the model. 
# 
# If you understand how to structure the input and output of the model, and know the fundamental concepts in machine learning, then just a high-level understanding of how an RNN works is sufficient for using Keras. You'll see that most of the code here is actually just data manipulation, and I'll visualize each step in this process. I'm focusing on this because when I was first learning about NLP, I felt like I lacked a basic understanding of how to represent and manipulate text data in code, maybe because it's assumed that it's trivial to figure out.
# 
# Even though it is not my focus here, it is more enlightening to understand the technical details of the RNN itself, and it's necessary if you want to innovate on it. For a better understanding of RNNs and neural networks in general, see the resources at the bottom of the notebook.
# 
# Here an RNN will be used to encode the text of a movie review, and this representation will be used to predict the numerical rating assigned by the reviewer. The model shown here can be applied to any task where the goal is to predict a numerical score associated with a piece of text. Hopefully you can substitute your own datasets and/or modify the code to adapt it to other tasks.
# 
# ### <font color='#6629b2'>Keras</font>
# 
# [Keras](https://keras.io/) is a Python deep learning framework that lets you quickly put together neural network models with a minimal amount of code. It can be run on top of [Theano](http://deeplearning.net/software/theano/) or [Tensor Flow](https://www.tensorflow.org/) without you needing to know either of these underlying frameworks. It provides implementations of several of the layer architectures, objective functions, and optimization algorithms you need for building a model.

# ## <font color='#6629b2'>Dataset</font>
# 
# The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) consists of 50,000 movie reviews from [IMDB](http://www.imdb.com/). The ratings are on a 1-10 scale, but the dataset only contains "polarized" reviews: positive reviews with a rating of 7 or higher, and negative reviews with a rating of 4 or lower. There are an equal number of positive and negative reviews. The reviews are divided into train and test sets with 25,000 reviews each.

# In[50]:

from __future__ import print_function #Python 2/3 compatibility for print statements


# I'll load the datasets using the [pandas library](https://pandas.pydata.org/), which is extremely useful for any task involving data storage and manipulation. This library puts a dataset into a readable table format, and makes it easy to retrieve specific columns and rows.

# In[55]:

'''Load the training dataset'''

import pandas

# For demo purposes, will load only the first 100 reviews in the training set
train_reviews = pandas.read_csv('dataset/imdb_train_reviews.csv', encoding='utf-8')[:100]
train_reviews[:10] #Show a sample


# ## <font color='#6629b2'>Preparing the data</font>

# ###  <font color='#6629b2'>Tokenization</font>
# 
# The first pre-processing step is to tokenize each of the reviews into (lowercased) individual words, since the RNN will encode the reviews word by word. For this I'll use [spacy](https://spacy.io/), which is a fast and extremely user-friendly library that performs various language processing tasks. 

# In[63]:

'''Split texts into lists of words (tokens)'''

import spacy

encoder = spacy.load('en')

def text_to_tokens(text_seqs):
    token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
    return token_seqs

train_reviews['Tokenized_Review'] = text_to_tokens(train_reviews['Review'])
    
train_reviews[['Review','Tokenized_Review']][:10]


# ###  <font color='#6629b2'>Lexicon</font>

# Then we need to assemble a lexicon (aka vocabulary) of words that the model needs to know. Each tokenized word in the reviews is added to the lexicon, and then each word is mapped to a numerical index that can be read by the model. Since large datasets may contain a huge number of unique words, it's common to filter all words occurring less than a certain number of times, and replace them with some generic &lt;UNK&gt; token. The min_freq parameter in the function below defines this threshold. In the example code, the min_freq parameter is set to 1, so the lexicon will contain all unique words in the training set. When assigning the indices, the number 1 will represent unknown words. The number 0 will represent "empty" word slots, which is explained below. Therefore "real" words will have indices of 2 or higher.

# In[116]:

'''Count tokens (words) in texts and add them to the lexicon'''

import pickle

def make_lexicon(token_seqs, min_freq=1):
    # First, count how often each word appears in the text.
    token_counts = {}
    for seq in token_seqs:
        for token in seq:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

    # Then, assign each word to a numerical index. Filter words that occur less than min_freq times.
    lexicon = [token for token, count in token_counts.items() if count >= min_freq]
    # Indices start at 2. 0 is reserved for padding, and 1 for unknown words.
    lexicon = {token:idx + 2 for idx,token in enumerate(lexicon)}
    lexicon[u'<UNK>'] = 1 # Unknown words are those that occur fewer than min_freq times
    lexicon_size = len(lexicon)

    pprint.pprint(list(lexicon.items())[:20])
    
    return lexicon

lexicon = make_lexicon(token_seqs=train_reviews['Tokenized_Review'], min_freq=10)
print("{} words in lexicon".format(len(lexicon)))

with open('pretrained_model/lexicon.pkl', 'wb') as f: # Save the lexicon by pickling it
    pickle.dump(lexicon, f)


# ###  <font color='#6629b2'>From strings to numbers</font>
# 
# Once the lexicon is built, we can transform each review from string tokens into a list of numerical indices.

# In[58]:

'''Convert each text from a list of tokens to a list of numbers (indices)'''

def tokens_to_idxs(token_seqs, lexicon):
    idx_seqs = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_seq]  
                                                                     for token_seq in token_seqs]
    return idx_seqs

train_reviews['Review_Idxs'] = tokens_to_idxs(token_seqs=train_reviews['Tokenized_Review'], 
                                              lexicon=lexicon)
                                   
train_reviews[['Tokenized_Review', 'Review_Idxs']][:10]


# ###  <font color='#6629b2'>Numerical lists to matrices</font>
# 
# We need to put all the reviews in the training set into a single matrix, where each row is a review and each column is a word index in that sequence. This enables the model to process multiple sequences in parallel (batches) as opposed to one at a time. Using batches significantly speeds up training. However, each review has a different number of words, so we create a padded matrix equal to the length on the longest review in the training set. For all reviews with fewer words, we prepend the row with zeros representing an empty word position. We can tell Keras to ignore these zeros during training.

# In[77]:

'''Create a padded matrix of input reviews'''

from keras.preprocessing.sequence import pad_sequences

def pad_idx_seqs(idx_seqs):
    max_seq_len = max([len(idx_seq) for idx_seq in idx_seqs]) # Get length of longest sequence
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len) # Keras provides a convenient padding function
    return padded_idxs

train_padded_idxs = pad_idx_seqs(train_reviews['Review_Idxs'])
print("length of input matrix: {}".format(train_padded_idxs.shape[-1]))

train_padded_idxs


# ##  <font color='#6629b2'>Building the model</font>
# 
# To assemble the model, we'll use Keras' [Functional API](https://keras.io/getting-started/functional-api-guide/), which is one of two ways to use Keras to assemble models (the alternative is the [Sequential API](https://keras.io/getting-started/sequential-model-guide/), which is a bit simpler but has more constraints). A model consists of a series of layers. As shown in the code below, we initialize instances for each layer. Each layer can be called with another layer as input, e.g. Embedding()(input_layer). A model instance is initialized with the Model() object, which defines the initial input and final output layers for that model. Before the model can be trained, the compile() function must be called with the loss function and optimization algorithm specified (see below).
# 
# ###  <font color='#6629b2'>Layers</font>
# 
# We'll build an RNN with three layers:
# 
# **1. Embedding**: An input [layer](https://keras.io/layers/embeddings/) that converts word indices into distributed vector representations (embeddings). The mask_zero=True parameter indicates that values of 0 in the matrix (the padding) will be ignored by the model.
# 
# **2. GRU**: A [recurrent (GRU) hidden layer](https://keras.io/layers/recurrent/), the central component of the model. As it observes each word in the story, it integrates the word embedding representation with what it's observed so far to compute a representation (hidden state) of the review at that timepoint. There are a few architectures for this layer - I use the GRU variation, Keras also provides LSTM or just the simple vanilla recurrent layer (see the materials at the bottom for an explanation of the difference). This layer outputs the last hidden state of the sequence (i.e. the hidden representation of the review after its last word is observed).
# 
# **3. Dense**: An output [layer](https://keras.io/layers/core/#dense) that predicts the rating for the review based on its GRU representation given by the previous layer. This output is continuous (i.e. ranging from 1-10) rather than categorical. The model gets feedback during training about what the actual rating should be.
# 
# The term "layer" is just an abstraction, when really all these layers are just matrices. The "weights" that connect the layers are also matrices. The process of training a neural network is a series of matrix multiplications. The weight matrices are the values that are adjusted during training in order for the model to learn to predict ratings. 
# 
# ###  <font color='#6629b2'>Parameters</font>
# 
# Our function for creating the model takes the following parameters:
# 
# **n_input_nodes**: the number of unique words in the lexicon, plus one to account for the padding represented by 0 values. This indicates the number of rows in the embedding layer, where each row corresponds to a word.
# 
# **n_embedding_nodes**: the number of dimensions (units) in the embedding layer, which can be freely defined. Here, it is set to 300.
# 
# **n_hidden_nodes**: the number of dimensions in the hidden layers. Like the embedding layer, this can be freely chosen. Here, it is set to 500.
# 
# ###  <font color='#6629b2'>Procedure</font>
# 
# The output of the model is a single continuous value (the predicted rating), making this a regression rather than a classification model. There is only one dimension in the output layer, which contains the predicted rating. Like all neural networks, RNNs learn by updating the parameters (weights) to optimize an objective (loss) function. For this model, the objective is to minimize the mean squared error between the predicted ratings and the actual ratings for the training reviews, thus bringing the predicted ratings closer to the real ratings. The details of this process are extensive; see the resources at the bottom of the notebook if you want a deeper understanding. One huge benefit of Keras is that it implements many of these details for you. Not only does it already have implementations of the types of layer architectures, it also has many of the [loss functions](https://keras.io/losses/) and [optimization methods](https://keras.io/optimizers/) you need for training various models.

# In[114]:

'''Create the model'''

from keras.models import Model
from keras.layers import Input, Dense
# from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_model(n_input_nodes, n_embedding_nodes, n_hidden_nodes):
    
    # Layer 1
    input_layer = Input(shape=(None,)) #Length of input matrix will be inferred from input, so None can be given as placeholder
    
    # Layer 2
    embedding_layer = Embedding(input_dim=n_input_nodes,
                                output_dim=n_embedding_nodes,
                                mask_zero=True)(input_layer)
    
    # Layer 3
    gru_layer = GRU(units=n_hidden_nodes)(embedding_layer)
    
    #Layer 4
    output_layer = Dense(units=1)(gru_layer)
    
    #Specify which layers are input and output, compile model with loss and optimization functions
    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(loss="mean_squared_error", optimizer='adam')
    
    return model

model = create_model(n_input_nodes=len(lexicon) + 1, n_embedding_nodes=300, n_hidden_nodes=500)


# ###  <font color='#6629b2'>Training</font>
# 
# Now we're ready to train the model. Keras' training function lets us specify the batch size and number of times to iterate through the training data (epochs). Keras reports the mean squared error loss after each epoch - if the model is learning correctly, it should progressively decrease.

# In[71]:

'''Train the model'''

model.fit(x=train_padded_idxs, y=train_reviews['Rating'], batch_size=50, epochs=10)
model.save('pretrained_model/model.h5') #save parameters of model 


# ##  <font color='#6629b2'>Predicting ratings for reviews</font>
# 
# Once the model is trained, we can use it predict the ratings for the reviews in the test set. To demonstrate this, I'll load a saved model previously trained on all the reviews in the training set.

# In[74]:

'''Load saved model'''

# Load lexicon
with open('pretrained_model/lexicon.pkl', 'rb') as f:
    lexicon = pickle.load(f)

# Load RNN model
from keras.models import load_model
model = load_model('pretrained_model/model.h5')


# In[78]:

'''Load the test dataset, tokenize, and transform to numerical matrix'''

test_reviews = pandas.read_csv('dataset/imdb_test_reviews.csv', encoding='utf-8')[:100]
test_reviews['Tokenized_Review'] = text_to_tokens(test_reviews['Review'])
test_reviews['Review_Idxs'] = tokens_to_idxs(token_seqs=test_reviews['Tokenized_Review'],
                                             lexicon=lexicon)
test_padded_idxs = pad_idx_seqs(test_reviews['Review_Idxs'])


# Then we can call the predict() function on the test reviews to get the predicted ratings.

# In[109]:

'''Show predicted ratings for test reviews'''

#Since ratings are integers, need to round predicted rating to nearest integer
test_reviews['Pred_Rating'] = numpy.round(model.predict(test_padded_idxs)[:,0]).astype(int)
test_reviews[['Review', 'Rating', 'Pred_Rating']]


# ### <font color='#6629b2'>Visualizing inner layers</font>
# 
# To help visualize the data representation inside the model, we can look at the output of each layer individually. Keras' Functional API lets you derive a new model with the layers from an existing model, so you can define the output to be a layer below the output layer in the original model. Calling predict() on this new model will produce the output of that layer for a given input. Of course, glancing at the numbers by themselves doesn't provide any interpretation of what the model has learned (although there are opportunities to [interpret these values](https://www.civisanalytics.com/blog/interpreting-visualizing-neural-networks-text-processing/)), but seeing them verifies the model is just a series of transformations from one matrix to another. 

# In[110]:

'''Show the output of embedding layer'''

embedding_layer = Model(inputs=model.layers[0].input, outputs=model.layers[1].output)
embedding_output = embedding_layer.predict(test_padded_idxs)
print("WORD EMBEDDINGS OUTPUT SHAPE:", embedding_output.shape)
print(embedding_output[0]) # Print embedding vectors for first review in test set


# In[112]:

'''Show the output of recurrent (GRU) layer'''

hidden_layer = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
hidden_output = hidden_layer.predict(test_padded_idxs)
print("HIDDEN LAYER OUTPUT SHAPE:", hidden_output.shape)
print(hidden_output[0]) # Print hidden vector for first review in test set


# ### <font color='#6629b2'>Evaluation</font>
# 
# A common evaluation for regression models like this one is $R^2$, called the the coefficient of determination. This metric indicates the proportion of variance in the output variable (the rating) that is predictable from the input variable (the review text). The best possible score is 1.0, which indicates the model always predicts the correct rating. The scikit-learn library provides several [evaluation metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) including $R^2$.

# In[113]:

'''Evaluate the model with R^2'''

from sklearn.metrics import r2_score

r2 = r2_score(y_true=test_reviews['Rating'], y_pred=test_pred_ratings)
print("COEFFICIENT OF DETERMINATION (R2): {:3f}".format(r2))


# ## <font color='#6629b2'>Conclusion</font>
# 
# As mentioned above, the model shown here could be applied to any task where the goal is to predict a score for a particular sequence. For ratings prediction, this score is ordinal, but it could also be categorical with a few simple changes to the output layer of the model. My other tutorials for [language modeling/generation](https://github.com/roemmele/keras-rnn-demo/language-modeling) and [part-of-speech tagging](https://github.com/roemmele/keras-rnn-demo/pos-tagging) demonstrate this type of prediction with categorical variables. They also show how to build an RNN in Keras when the output is a sequence of labels, rather than a single value as shown here.

# ## <font color='#6629b2'>More resources</font>
# 
# Yoav Goldberg's book [Neural Network Methods for Natural Language Processing](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) is a thorough introduction to neural networks for NLP tasks in general.
# 
# If you'd like to learn more about what Keras is doing under the hood, there is a [Theano tutorial](http://deeplearning.net/tutorial/lstm.html) that also applies an RNN to sentiment prediction, using the same dataset here
# 
# Andrej Karpathy's blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is very helpful for understanding the mathematical details of an RNN, applied to the task of language modeling. It also provides raw Python code with an implementation of the backpropagation algorithm.
# 
# TensorFlow also has an RNN language model [tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/recurrent/index.html) using the Penn Treebank dataset
# 
# Chris Olah provides a good [explanation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of how LSTM RNNs work (this explanation also applies to the GRU model used here)
# 
# Denny Britz's [tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) documents well both the technical details of RNNs and their implementation in Python.
