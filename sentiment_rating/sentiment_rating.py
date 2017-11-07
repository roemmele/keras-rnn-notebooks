
# coding: utf-8

# # <font color='#6629b2'>Predicting sentiment ratings with recurrent neural networks using Keras</font>
# ### https://github.com/roemmele/keras-rnn-notebooks
# by Melissa Roemmele, 10/23/17, roemmele @ ict.usc.edu

# ## <font color='#6629b2'>Overview</font>
# 
# I am going to show how to use the Keras library to build a recurrent neural network (RNN) model that predicts sentiment ratings for text sequences. Specifically, the model will predict the ratings associated with movie reviews.
# 
# ### <font color='#6629b2'>Recurrent Neural Networks</font>
# 
# RNNs are a general framework for modeling sequence data and are particularly useful for natural language processing tasks. At a high level, RNN encode sequences via a set of parameters (weights) that are optimized to predict some output variable. The notebook demonstrates the code needed to assemble an RNN model using the Keras library, as well as some data processing tools that facilitate building the model. 
# 
# If you understand how to structure the input and output of the model, and know the fundamental concepts in machine learning, then a high-level understanding of how an RNN works is sufficient for using Keras. You'll see that most of the code here is actually just data manipulation, and I'll visualize each step in this process. The code used to assemble the RNN itself is more minimal. It is of course useful to know these details, so you can theorize on the results and innovate the model to make it better. For a better understanding of RNNs and neural networks in general, see the resources at the bottom of the notebook.
# 
# Here an RNN will be used to encode the text of a movie review, and this representation will be used to predict the numerical rating assigned by the reviewer. The model shown here can be applied to any task where the goal is to predict a numerical score associated with a piece of text. Hopefully you can substitute your own datasets and/or modify the code to adapt it to other tasks.
# 
# ### <font color='#6629b2'>Keras</font>
# 
# [Keras](https://keras.io/) is a Python deep learning framework that lets you quickly put together neural network models with a minimal amount of code. It can be run on top of the mathematical optimization libraries [Theano](http://deeplearning.net/software/theano/) or [Tensor Flow](https://www.tensorflow.org/) without you needing to know either of these underlying frameworks. It provides implementations of several of the layer architectures, objective functions, and optimization algorithms you need for building a model.

# ## <font color='#6629b2'>Dataset</font>
# 
# The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) consists of 50,000 movie reviews from [IMDB](http://www.imdb.com/). The ratings are on a 1-10 scale, but the dataset only contains "polarized" reviews: positive reviews with a rating of 7 or higher, and negative reviews with a rating of 4 or lower. There are an equal number of positive and negative reviews. In the full dataset, the reviews are divided into train and test sets with 25,000 reviews each. Here I'm just going to load a sample training set of 100 reviews, so you can download the full dataset at the above link.

# In[ ]:

from __future__ import print_function #Python 2/3 compatibility for print statements
import pandas
pandas.set_option('display.max_colwidth', 170) #widen pandas rows display


# I'll load the datasets using the [pandas library](https://pandas.pydata.org/), which is extremely useful for any task involving data storage and manipulation. This library puts a dataset into a readable table format, and makes it easy to retrieve specific columns and rows.

# In[ ]:

'''Load the training dataset'''

train_reviews = pandas.read_csv('dataset/example_train_imdb_reviews.csv', encoding='utf-8')
train_reviews[:10]


# ## <font color='#6629b2'>Preparing the data</font>
# 
# ###  <font color='#6629b2'>Tokenization</font>
# 
# The first pre-processing step is to tokenize each of the reviews into (lowercased) individual words, since the RNN will encode the reviews word by word. For this I'll use [spaCy](https://spacy.io/), which is a fast and extremely user-friendly library that performs various language processing tasks. 

# In[ ]:

'''Split texts into lists of words (tokens)'''

import spacy

encoder = spacy.load('en')

def text_to_tokens(text_seqs):
    token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
    return token_seqs

train_reviews['Tokenized_Review'] = text_to_tokens(train_reviews['Review'])
    
train_reviews[['Review','Tokenized_Review']][:10]


# ###  <font color='#6629b2'>Lexicon</font>

# Then we need to assemble a lexicon (aka vocabulary) of words that the model needs to know. Each tokenized word in the reviews is added to the lexicon, and then each word is mapped to a numerical index that can be read by the model. Since large datasets may contain a huge number of unique words, it's common to filter all words occurring less than a certain number of times, and replace them with some generic &lt;UNK&gt; token. The min_freq parameter in the function below defines this threshold. When assigning the indices, the number 1 will represent unknown words. The number 0 will represent "empty" word slots, which is explained below. Therefore "real" words will have indices of 2 or higher.

# In[ ]:

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

    print("LEXICON SAMPLE ({} total items):".format(len(lexicon)))
    print(list(lexicon.items())[:20])
    
    return lexicon

lexicon = make_lexicon(token_seqs=train_reviews['Tokenized_Review'], min_freq=1)

with open('example_model/lexicon.pkl', 'wb') as f: # Save the lexicon by pickling it
    pickle.dump(lexicon, f)


# ###  <font color='#6629b2'>From strings to numbers</font>
# 
# Once the lexicon is built, we can use it to transform each review from a list of string tokens into a list of numerical indices.

# In[ ]:

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

# In[ ]:

'''Create a padded matrix of input reviews'''

from keras.preprocessing.sequence import pad_sequences

def pad_idx_seqs(idx_seqs):
    max_seq_len = max([len(idx_seq) for idx_seq in idx_seqs]) # Get length of longest sequence
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len) # Keras provides a convenient padding function
    return padded_idxs

train_padded_idxs = pad_idx_seqs(train_reviews['Review_Idxs'])

print("TRAIN INPUT:\n", train_padded_idxs)
print("SHAPE:", train_padded_idxs.shape, "\n")


# ##  <font color='#6629b2'>Building the model</font>
# 
# To assemble the model, we'll use Keras' [Functional API](https://keras.io/getting-started/functional-api-guide/), which is one of two ways to use Keras to assemble models (the alternative is the [Sequential API](https://keras.io/getting-started/sequential-model-guide/), which is a bit simpler but has more constraints). A model consists of a series of layers. As shown in the code below, we initialize instances for each layer. Each layer can be called with another layer as input, e.g. Embedding()(input_layer). A model instance is initialized with the Model() object, which defines the initial input and final output layers for that model. Before the model can be trained, the compile() function must be called with the loss function and optimization algorithm specified (see below).
# 
# ###  <font color='#6629b2'>Layers</font>
# 
# We'll build an RNN with four layers:
# 
# **1. Input**: The input layer takes in the matrix of word indices.
# 
# **2. Embedding**: A [layer](https://keras.io/layers/embeddings/) that converts integer word indices into distributed vector representations (embeddings). The mask_zero=True parameter indicates that values of 0 in the matrix (the padding) will be ignored by the model.
# 
# **3. GRU**: A [recurrent (GRU) hidden layer](https://keras.io/layers/recurrent/), the central component of the model. As it observes each word in the story, it integrates the word embedding representation with what it's observed so far to compute a representation (hidden state) of the review at that timepoint. There are a few architectures for this layer - I use the GRU variation, Keras also provides LSTM or just the simple vanilla recurrent layer (see the materials at the bottom for an explanation of the difference). This layer outputs the last hidden state of the sequence (i.e. the hidden representation of the review after its last word is observed).
# 
# **4. Dense**: An output [layer](https://keras.io/layers/core/#dense) that predicts the rating for the review based on its GRU representation given by the previous layer. This output is continuous (i.e. ranging from 1-10) rather than categorical. The model gets feedback during training about what the actual rating should be.
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
# The output of the model is a single continuous value (the predicted rating), making this a regression rather than a classification model. There is only one dimension in the output layer, which contains the predicted rating. Like all neural networks, RNNs learn by updating the parameters (weights) to optimize an objective (loss) function. For this model, the objective is to minimize the mean squared error between the predicted ratings and the actual ratings for the training reviews, thus bringing the predicted ratings closer to the real ratings. The details of this process are extensive; see the resources at the bottom of the notebook if you want a deeper understanding. One huge benefit of Keras is that it implements many of these details for you. Not only does it already have implementations of the types of layer architectures, it also has many of the [loss functions](https://keras.io/losses/) and [optimization methods](https://keras.io/optimizers/) you need for training various models. The specific loss function and optimization method you use is specified when compiling the model with the model.compile() function.

# In[ ]:

'''Create the model'''

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_model(n_input_nodes, n_embedding_nodes, n_hidden_nodes):
    
    # Layer 1 -  Technically the shape of this layer is (batch_size, len(train_padded_idxs)).
    # However, both the batch size and the length of the input matrix can be inferred from the input at training time. 
    # The batch size is implictly included in the shape of the input, so it does not need to 
    # be specified as a dimension of the input. None can be given as placeholder for the input matrix length.
    # By defining it as None, the model is flexible in accepting inputs with different lengths.
    input_layer = Input(shape=(None,))
    
    # Layer 2
    embedding_layer = Embedding(input_dim=n_input_nodes,
                                output_dim=n_embedding_nodes,
                                mask_zero=True)(input_layer) #mask_zero tells the model to ignore 0 values (padding)
    #Output shape = (batch_size, input_matrix_length, n_embedding_nodes)
    
    # Layer 3
    gru_layer = GRU(units=n_hidden_nodes)(embedding_layer)
    #Output shape = (batch_size, n_hidden_nodes)
    
    #Layer 4
    output_layer = Dense(units=1)(gru_layer)
    #Output shape = (batch_size, 1)
    
    #Specify which layers are input and output, compile model with loss and optimization functions
    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(loss="mean_squared_error", optimizer='adam')
    
    return model


# In[ ]:

model = create_model(n_input_nodes=len(lexicon) + 1, n_embedding_nodes=300, n_hidden_nodes=500)


# ###  <font color='#6629b2'>Training</font>
# 
# Now we're ready to train the model. Keras will apply batch training by default, even though we didn't specify the batch size when creating the model. If a batch size isn't given, Keras will use its default (32). The training function also indicates the number of times to iterate through the training data (epochs). Keras reports the mean squared error loss after each epoch - if the model is learning correctly, it should progressively decrease.

# In[ ]:

'''Train the model'''

model.fit(x=train_padded_idxs, y=train_reviews['Rating'], batch_size=20, epochs=5)
model.save('example_model/model.h5') #save parameters of model 


# ##  <font color='#6629b2'>Predicting ratings for reviews</font>
# 
# Once the model is trained, we can use it predict the ratings for the reviews in the test set. To demonstrate this, I'll load a saved model previously trained on all 25,000 reviews in the training set. I'll apply this model to an example test set of 100 reviews (again, there are 25,000 reviews in the full test set provided at the above link).

# In[ ]:

'''Load saved model'''

# Load lexicon
with open('pretrained_model/lexicon.pkl', 'rb') as f:
    lexicon = pickle.load(f)

# Load RNN model
from keras.models import load_model
model = load_model('pretrained_model/model.h5')


# In[ ]:

'''Load the test dataset, tokenize, and transform to numerical matrix'''

test_reviews = pandas.read_csv('dataset/example_test_imdb_reviews.csv', encoding='utf-8')
test_reviews['Tokenized_Review'] = text_to_tokens(test_reviews['Review'])
test_reviews['Review_Idxs'] = tokens_to_idxs(token_seqs=test_reviews['Tokenized_Review'],
                                             lexicon=lexicon)
test_padded_idxs = pad_idx_seqs(test_reviews['Review_Idxs'])

print("TEST INPUT:\n", test_padded_idxs)
print("SHAPE:", test_padded_idxs.shape, "\n")


# Then we can call the predict() function on the test reviews to get the predicted ratings.

# In[ ]:

'''Show predicted ratings for test reviews'''

import numpy

#Since ratings are integers, need to round predicted rating to nearest integer
test_reviews['Pred_Rating'] = numpy.round(model.predict(test_padded_idxs)[:,0]).astype(int)
test_reviews[['Review', 'Rating', 'Pred_Rating']]


# ### <font color='#6629b2'>Evaluation</font>
# 
# A common evaluation for regression models like this one is $R^2$, called the the coefficient of determination. This metric indicates the proportion of variance in the output variable (the rating) that is predictable from the input variable (the review text). The best possible score is 1.0, which indicates the model always predicts the correct rating. The scikit-learn library provides several [evaluation metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) for machine learning models, including $R^2$.

# In[ ]:

'''Evaluate the model with R^2'''

from sklearn.metrics import r2_score

r2 = r2_score(y_true=test_reviews['Rating'], y_pred=test_reviews['Pred_Rating'])
print("COEFFICIENT OF DETERMINATION (R2): {:3f}".format(r2))


# ### <font color='#6629b2'>Visualizing data inside the model</font>
# 
# To help visualize the data representation inside the model, we can look at the output of each layer individually. Keras' Functional API lets you derive a new model with the layers from an existing model, so you can define the output to be a layer below the output layer in the original model. Calling predict() on this new model will produce the output of that layer for a given input. Of course, glancing at the numbers by themselves doesn't provide any interpretation of what the model has learned (although there are opportunities to [interpret these values](https://www.civisanalytics.com/blog/interpreting-visualizing-neural-networks-text-processing/)), but seeing them verifies the model is just a series of transformations from one matrix to another. The model stores its layers as the list model.layers, and you can retrieve specific layer by its position index in the model.

# In[ ]:

'''Show the output of the embedding layer (second layer) for the test reviews'''

embedding_layer = Model(inputs=model.layers[0].input, outputs=model.layers[1].output) #embedding layer is 2nd layer (index 1)
embedding_output = embedding_layer.predict(test_padded_idxs)
print("EMBEDDING LAYER OUTPUT SHAPE:", embedding_output.shape)
print(embedding_output[0])


# It is also easy to look at the weight matrices that connect the layers. The get_weights() function will show the incoming weights for a particular layer.

# In[ ]:

'''Show weights that connect the hidden layer to the output layer (final layer)'''

hidden_to_output_weights = model.layers[-1].get_weights()[0]
print("HIDDEN-TO_OUTPUT WEIGHTS SHAPE:", hidden_to_output_weights.shape)
print(hidden_to_output_weights)


# ## <font color='#6629b2'>Conclusion</font>
# 
# As mentioned above, the model shown here could be applied to any task where the goal is to predict a score for a particular sequence. For ratings prediction, this score is ordinal, but it could also be categorical with a few simple changes to the output layer of the model. My other notebooks in this repository for language modeling/generation and part-of-speech tagging demonstrate this type of prediction with categorical variables. They also show how to build an RNN in Keras when the output is a sequence of labels, rather than a single value as shown here.

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
