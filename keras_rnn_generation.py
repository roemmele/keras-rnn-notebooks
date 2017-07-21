
# coding: utf-8

# # <font color='#6629b2'>Generating text with recurrent neural networks using Keras</font>
# by Melissa Roemmele, 7/17/17, roemmele @ usc.edu
# 
# ## <font color='#6629b2'>Overview</font>
# 
# I am going to show how to build a recurrent neural network (RNN) language model that learns the relation between words in text, using the Keras library for machine learning. I will then show how this model can be used for text generation.

# ## <font color='#6629b2'>Recurrent Neural Networks (RNNs)</font>
# 
# RNNs are a general framework for modeling sequence data and are particularly useful for natural langugage processing tasks. Here an RNN will be used as a language model, which can predict which word is likely to occur next in a text given the words before it.

# ## <font color='#6629b2'>Keras</font>
# 
# [Keras](https://keras.io/) is a Python deep learning framework that lets you quickly put together neural network models with a minimal amount of code. It can be run on top of [Theano](http://deeplearning.net/software/theano/) or [Tensor Flow](https://www.tensorflow.org/) without you needing to know either of these underlying frameworks. It provides implementations of several of the layer architectures, objective functions, and optimization algorithms you need for building a model.

# ## <font color='#6629b2'>Dataset</font>
# 
# My research is on story generation, so I've selected a dataset of stories as the text to be modeled by the RNN. They come from the [ROCStories](http://cs.rochester.edu/nlp/rocstories/) dataset, which consists of thousands of five-sentence stories about everyday life events. Here the model will observe all five sentences in each story. Then we'll use the trained model to generate the final sentence in a set of stories not observed during training.

# In[7]:

from __future__ import print_function #Python 2/3 compatibility for print statements
import pprint #pretty printing


# In[9]:

'''load the training dataset'''
import csv

with open('example_train_stories.csv', 'r') as f:
    train_stories = [story for story in csv.reader(f)]
    
#sentences in stories are comma-separated, so join them
train_stories = [" ".join(story) for story in train_stories]
pprint.pprint(train_stories[:2])


# ## <font color='#6629b2'>Preparing the data</font>
# 
# The model we'll create is a word-based language model, which means each input unit is a single word (some language models learn subword units like characters). 
# 
# So first we need to tokenize each of the stories into (lowercased) individual words. I'll use Keras' built-in tokenizer here for convenience, but typically I like to use [spacy](https://spacy.io/), a fast and user-friendly library that performs various language processing tasks. 
# 
# A note: Keras' tokenizer does not do the same linguistic processing to separate punctuation from words, for instance, which should be their own tokens. You can see this below from words that end in punctuation like "." or ",".
# 
# We need to assemble a lexicon (aka vocabulary) of words that the model needs to know. Thus, each tokenized word in the stories is added to the lexicon. We use the fit_on_texts() function to map each word in the stories to a numerical index. When working with large datasets it's common to filter all words occurring less than a certain number of times, and replace them with some generic "UNKNOWN" token. Here, because this dataset is small, every word encountered in the stories is added to the lexicon.

# In[10]:

'''make the lexicon'''

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(train_stories) #split stories into words, assign number to each unique word
pprint.pprint(list(tokenizer.word_index.items())[:20])

import pickle
with open('example_tokenizer.pkl', 'wb') as f: #save the tokenizer
    pickle.dump(tokenizer, f)


# In[11]:

'''convert each story from text to numbers'''

train_idxs = tokenizer.texts_to_sequences(train_stories) #transform each word to its numerical index in lexicon
pprint.pprint(train_stories[0])
pprint.pprint(train_idxs[0]) #show example of encoded story


# ###  <font color='#6629b2'>Creating a matrix</font>
# 
# Finally, we need to put all the training stories into a single matrix, where each row is a story and each column is a word index in that story. This enables the model to process the stories in batches as opposed to one at a time, which significantly speeds up training. However, each story has a different number of words. So we create a padded matrix equal to the length on the longest story in the training set. For all stories with fewer words, we prepend the row with zeros representing an empty word position. Then we can actually tell Keras to ignore these zeros during training.

# In[12]:

'''create a padded matrix of stories'''

from keras.preprocessing.sequence import pad_sequences

maxlen = max([len(story) for story in train_idxs]) # get length of longest story
print("matrix length:", maxlen)

train_idxs = pad_sequences(train_idxs, maxlen=maxlen) #keras provides convenient padding function
pprint.pprint(train_idxs[0]) #same example story as above


# ### <font color='#6629b2'>Defining the input and output</font>
# 
# In an RNN language model, the data is set up so that each word in the text is mapped to the word that follows it. In a given story, for each input word x[idx], the output label y[idx] is just x[idx+1].

# In[13]:

'''set up the model input and output'''

train_x = train_idxs[:, :-1]
print("x:")
pprint.pprint(train_x[0])
    
train_y = train_idxs[:, 1:]#, None] #Keras requires extra dim for y: (batch_size, n_timesteps, 1)
print("y:")
pprint.pprint(train_y[0])


# ##  <font color='#6629b2'>Creating the model</font>
# 
# We'll build an RNN with four layers: 
# 1. An input layer that converts word indices into distributed vector representations (embeddings).
# 2. A recurrent hidden layer, the main component of the network. As it observes each word in the story, it integrates the word embedding representation with what it's observed so far to compute a representation (hidden state) of the story at that timepoint. There are a few architectures for this layer - I use the GRU variation, Keras also provides LSTM or just the simple vanilla recurrent layer.
# 3. A second recurrent layer that takes the first as input and operates the same way, since adding more layers generally improves the model.
# 3. A prediction (dense) layer that outputs a probability for each word in the lexicon via the softmax function, where each probability indicates the chance of that word being the next word in the sequence. The model gets feedback during training about what the actual word should be.
# 
# Of course this is a very simplified explanation of the model, since the focus here is on how to implement it in Keras. For a more thorough explanation of RNNs, see the resources at the bottom of the notebook.
# 
# For each layer, we need to specify the number of dimensions (units). For the embedding and recurrent layers, this number can be freely defined (it is typically between 50-1000). For the output (prediction) layer, the number of units is equal to the lexicon size, since the model computes a probability distribution for each word in the lexicon. To account for the zeros in the input, we'll add one more dimension so that each word index corresponds to its output dimension, i.e. the predicted probability of word index 1 is at column index 1 (2nd column) in the probability distribution output.
# 
# When setting up the model, we specify the number of stories in each input batch (batch size) as well as the number of words in each story (n_timesteps). Here, we'll set n_timesteps to be the length of the x and y matrices above.\**  So the shape of the input to the model is (batch_size, n_timesteps). The embedding layer needs to be told how many unique word indices there are (input_dim=lexicon size + 1, adding one since the 0 index is reserved for padding) so that it can map each word to a vector of size output_dim=n_embedding_nodes. Thus the shape of the embedding layer output will be (batch_size, n_timesteps, n_embedding_nodes).
# 
# In the recurrent layers, return_sequences=True indicates the hidden state for each word in the story will be returned, as opposed to just the hidden state for the last word. This is necessary for the model to provide an output for each word. The stateful=True setting indicates the RNN will "remember" its hidden state until it is explicitly told to forget it via the reset_states() function. This comes into play during the generation stage (or also when n_timesteps is less than the length of x and y\**), so I will explain this further below.
# 
# For each word in a story, the prediction layer will output a probability distribution for the next word. To get this sequence of probability distributions rather than just one, we wrap TimeDistributed() class around the Dense layer. The model is trained to maximize the probabilities of the words in the stories, which is what the sparse_categorical_crossentropy loss function does (again, see below for a full explanation of this). 
# 
# One huge benefit of Keras is that it has several optimization algorithms already implemented. I use Adam here, there are several other available including SGD, RMSprop, and Adagrad. You can change other parameters like learning rate and gradient clipping as well.
# 
# *\**It is also possible to set n_timesteps to be less than this length and iterate over shorter sequences of words. For example, if we set n_timesteps to 10, the model will slide over each window of 10 words in the stories and perform an update to the parmaters by backpropogating the gradient over these 10 words (for the details of backpropogation, see below). However, we still want the model to "remember" everything in the story, not just the previous 10 words, so Keras provides the "stateful" option to do this. By setting "stateful=True" (here is it False), the hidden state of the model after observing 10 words will be carried over to the next word window. After all the words in a batch of stories have been processed, the reset_states() function can be called to indicate the model should now forget its hidden state and start over with the next batch of stories. You'd need to update the training function below to iterate through a batch of stories by n_timesteps at a time.*

# In[14]:

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_rnn(lexicon_size, n_embedding_nodes, n_hidden_nodes, batch_size, n_timesteps):

    rnn = Sequential()

    #Layer 1
    embedding_layer = Embedding(batch_input_shape=(batch_size, n_timesteps),
                                input_dim=lexicon_size + 1, #add 1 because word indices start at 1, not 0
                                output_dim=n_embedding_nodes, 
                                mask_zero=True) #mask_zero=True will ignore padding
    rnn.add(embedding_layer) #output shape is (batch_size, n_timesteps, n_embedding_nodes)

    #Layer 2
    recurrent_layer1 = GRU(n_hidden_nodes,
                           return_sequences=True, #return hidden state for each word, not just last one
                           stateful=True) #keep track of hidden state while iterating through story
    rnn.add(recurrent_layer1) #output shape is (batch_size, n_timesteps, n_hidden_nodes)

    #Layer 3
    recurrent_layer2 = GRU(n_hidden_nodes,
                           return_sequences=True, 
                           stateful=True)
    rnn.add(recurrent_layer2)  #output shape is (batch_size, n_timesteps, n_hidden_nodes)

    #Layer 4
    prediction_layer = TimeDistributed(Dense(lexicon_size + 1,
                                       activation="softmax"))
    rnn.add(prediction_layer) #output shape is (batch_size, n_timesteps, lexicon_size + 1)

    #Specify loss function and optimization algorithm, compile model
    rnn.compile(loss="sparse_categorical_crossentropy", 
                optimizer='adam')
    
    return rnn


# We'll create an RNN with 300 embedding nodes and 500 hidden nodes in each recurrent layer, with a batch size of 20 stories.

# In[15]:

'''initialize the RNN'''

batch_size = 20
rnn = create_rnn(lexicon_size = len(tokenizer.word_index),
                 n_embedding_nodes = 300,
                 n_hidden_nodes = 500,
                 batch_size = batch_size,
                 n_timesteps = maxlen - 1) #subtract 1 from maxlen because x and y each have one word less


# We'll train the RNN for 10 iterations through the training stories (epochs). The cross-entropy loss indicates how well the model is learning - it should go down with each epoch.

# In[16]:

'''train the RNN'''

import numpy

n_epochs = 10
print("Training RNN on", len(train_stories), "stories for", n_epochs, "epochs...")
for epoch in range(n_epochs):
    losses = []  #track cross-entropy loss during training
    for batch_idx in range(0, len(train_stories), batch_size):
        batch_x = train_x[batch_idx:batch_idx+batch_size] #get batch for x
        batch_y = train_y[batch_idx:batch_idx+batch_size, :, None] #Keras requires y shape:(batch_size, y_length, 1)
        loss = rnn.train_on_batch(batch_x, batch_y) #takes a few moments to initialize training
        losses.append(loss)
        rnn.reset_states() #reset hidden state after each batch
    print("epoch", epoch + 1, "mean loss: %.3f" % numpy.mean(losses))
    rnn.save('example_rnn.h5') #save model after each epoch


# ## <font color='#6629b2'>Generating sentences</font>
# 
# Now that the model is trained, it can be used to generate new text\**. Here, I'll give the model the first four sentences of a new story and have it generate the fifth sentence. To do this, the model reads the initial story in order to produce a probability distribution for the first word in the fifth sentence. We can sample a word from this probability distribution and add it to the story. We repeat this process, each time generating the next word based on the story so far. We stop generating words either when an end-of-sentence token is generated (e.g. ".", "!", or "?"). Of course, you can define any stopping criteria (e.g. a specific number of words). 
# 
# *\**Since the above code takes awhile to run, here I'm going to load a pre-trained model (rnn_96000.h5 and the accompanying tokenizer_96000.pkl) that was trained on 96,000 stories in this corpus for 25 epochs, with the same model parameters shown above. Obviously you should substitute the file names for your trained model here.*

# In[17]:

'''load the trained model; in case training was skipped, load all libaries'''

import numpy, pickle, csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.models import load_model

with open('tokenizer_96000.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    print("loaded tokenizer with", len(tokenizer.word_index), "words in lexicon")

rnn = load_model('rnn_96000.h5')


# In[19]:

'''load stories used for generation'''

with open('example_test_stories.csv', 'r') as f:
    heldout_stories = [story for story in csv.reader(f)]

#separate final sentence from first four, which will be used for generate new final sentence
heldout_endings = [story[-1] for story in heldout_stories[-10:]]
heldout_stories = [" ".join(story[:-1]) for story in heldout_stories[-10:]]
heldout_idxs = tokenizer.texts_to_sequences(heldout_stories)
print("STORY:", heldout_stories[0], "\n", heldout_idxs[0], "\n")
print("GIVEN ENDING:", heldout_endings[0])


# The model will generate word indices, so we need to map these numbers back to their corresponding strings. We'll reverse the lexicon dictionary to create a lookup table to get each word from its index.

# In[20]:

'''create lookup table to get string words from their indices'''

lexicon_lookup = {index: word for word, index in tokenizer.word_index.items()}
eos_tokens = [".", "?", "!"] #specify which characters should indicate the end of a sentence and halt generation

pprint.pprint(list(lexicon_lookup.items())[:20]) #print a sample of the lookup table


# When generating, the model predicts one word at a time for a given story, but the trained model expects that batch size = 20 and n_timesteps = 63. The easiest thing to do is duplicate the trained model but set the batch size = 1 and n_timesteps = 1. To do this, we just create a new model with these settings and then copy the parameters (weights) of the trained model over the new model.

# In[21]:

'''duplicate the trained RNN but set batch size = 1 and n_timesteps = 1'''

generation_rnn = create_rnn(lexicon_size = len(tokenizer.word_index),
                            n_embedding_nodes = 300,
                            n_hidden_nodes = 500,
                            batch_size = 1,
                            n_timesteps = 1)
generation_rnn.set_weights(rnn.get_weights())


# Now we can iterate through each story and generate an ending for it. For each story, we need to "load" its first four sentences into the model. This can be done using predict_on_batch() function, even though the probability distributions returned by this function are not needed when just reading the story. Because we set stateful=True when creating the RNN, Keras will keep track of the hidden state while iterating through each word, so that's why n_timesteps can be set to 1. Once the ending has been generated, we call reset_states() to clear the hidden state so that the next story can be read.
# 
# Once the final word in the fourth sentence has been read in a given story, then we use the resulting probability distribution to predict the first word in the fifth sentence. We use numpy.random.choice() to select a word according to its probability. We once again call predict_on_batch() to get a probability distribution for the second word and sample from this distribution. We continue doing this until a word that ends with an end-of-sentence puncutation mark has been selected. Then we decode the generated ending into a string and show it next to the ending that was given in the dataset.
# 
# You can see that the generated endings are generally not as coherent and well-formed as the human-authored endings, but they do capture some components of the story and they are often more entertaining.
# 

# In[22]:

'''use RNN to generate new endings for stories'''

for story, story_idxs, ending in zip(heldout_stories, heldout_idxs, heldout_endings):
    print("STORY:", story)
    print("GIVEN ENDING:", ending)
    
    generated_ending = []
    
    story_idxs = numpy.array(story_idxs)[None] #format story with shape (1, length)
    
    for step_idx in range(story_idxs.shape[-1]):
        p_next_word = generation_rnn.predict_on_batch(story_idxs[:, step_idx])[0,-1] #load the story; input shape will be (1, 1)

    while not generated_ending or lexicon_lookup[next_word][-1] not in eos_tokens: #now start predicting new words
        next_word = numpy.random.choice(a=p_next_word.shape[-1], p=p_next_word)
        generated_ending.append(next_word)
        p_next_word = generation_rnn.predict_on_batch(numpy.array(next_word)[None,None])[0,-1]
    
    generation_rnn.reset_states() #reset hidden state after generating ending
    
    generated_ending = " ".join([lexicon_lookup[word] 
                                 for word in generated_ending]) #decode from numbers back into words
    print("GENERATED ENDING:", generated_ending, "\n")
    


# ## <font color='#6629b2'>Conclusion</font>
# 
# Because it's an amusing task and illustrates the power of RNNs, there are now many tutorials online about text generation with RNNs. This one shows one way to do it in Keras with batch training when the length of the sequences is variable. This also demonstrates how you can input existing text into the RNN and generate a continuation of it.
# 
# There are many ways this language model can be made to be more sophisticated. Here's a few interesting papers from the NLP community that innovate this basic model for different generation tasks:
# 
# *Recipe generation:* [Globally Coherent Text Generation with Neural Checklist Models](https://homes.cs.washington.edu/~yejin/Papers/emnlp16_neuralchecklist.pdf). Chloé Kiddon, Luke Zettlemoyer, and Yejin Choi. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.
# 
# *Emotional text generation:* [Affect-LM: A Neural Language Model for Customizable Affective Text Generation](https://arxiv.org/pdf/1704.06851.pdf). Sayan Ghosh, Mathieu Chollet, Eugene Laksana, Louis-Philippe Morency, Stefan Scherer. Annual Meeting of the Association for Computational Linguistics (ACL), 2017.
# 
# *Poetry generation:* [Generating Topical Poetry](https://www.isi.edu/natural-language/mt/emnlp16-poetry.pdf). Marjan Ghazvininejad, Xing Shi, Yejin Choi, and Kevin Knight. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.
# 
# *Dialogue generation:* [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://www-etud.iro.umontreal.ca/~sordonia/pdf/naacl15.pdf). Alessandro Sordoni, Michel Galley, Michael Auli, Chris Brockett, Yangfeng Ji, Margaret Mitchell, Jian-Yun Nie1, Jianfeng Gao, Bill Dolan. North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2015.

# ## <font color='#6629b2'>Helpful resources about RNNs for text processing</font>
# 
# Among the [Theano tutorials](http://deeplearning.net/tutorial/) mentioned above, there are two specifically on RNNs for NLP: [semantic parsing](http://deeplearning.net/tutorial/rnnslu.html#rnnslu) and [sentiment analysis](http://deeplearning.net/tutorial/lstm.html#lstm)
# 
# [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (same model as shown here, with raw Python code) 
# 
# TensorFlow also has an RNN language model [tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/recurrent/index.html) using the Penn Treebank dataset
# 
# This [explanation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of how LSTMs work and why they are better than plain RNNs (this explanation also applies to the GRU used here)
# 
# Another [tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) that documents well both the theory of RNNs and their implementation in Python (and if you care to implement the details of the stochastic gradient descent and backprogation through time algorithms, this is very informative)
