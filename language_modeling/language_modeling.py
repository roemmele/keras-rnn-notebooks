
# coding: utf-8

# # <font color='#6629b2'>Language modeling with recurrent neural networks using Keras</font>
# ### https://github.com/roemmele/keras-rnn-demo/language-modeling
# by Melissa Roemmele, 10/30/17, roemmele @ usc.edu
# 
# ## <font color='#6629b2'>Overview</font>
# 
# I am going to show how to build a recurrent neural network (RNN) language model that learns the relation between words in text, using the Keras library for machine learning. I will then show how this model can be used for text generation.
# 
# ### <font color='#6629b2'>Language Modeling</font>
# 
# A language model is a model of the probability of word sequences. These models are useful for a variety of tasks, such as ones that require selecting from a set of candiate outputs as in speech recognition or machine translation, for example. Here, I'll show how a language model can be used to generate the endings of stories. Language generation is a difficult research problem which is generally addressed by more complex models than the one shown here.
# 
# Traditionally, the most well-known approach to language modeling relies on n-grams. The limitation of n-gram language models is that they only explitly model the probability of a sequence of *n* words. In contrast, RNNs can model longer sequences and thus typically are better at predicting which words will appear in a sequence. See the [chapter in Jurafsky & Martin's *Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/4.pdf) to learn more about traditional approaches to language modeling. 
# 
# ### <font color='#6629b2'>Recurrent Neural Networks (RNNs)</font>
# 
# RNNs are a general framework for modeling sequence data and are particularly useful for natural language processing tasks. At a high level, RNN encode sequences via a set of parameters (weights) that are optimized to predict some output variable. The focus of this tutorial is on the code needed to assemble a model in Keras. For a more general introduction to RNNs, see the resources at the bottom. Here an RNN will be used as a language model, which can predict which word is likely to occur next in a text given the words before it.
# 
# ### <font color='#6629b2'>Keras</font>
# 
# [Keras](https://keras.io/) is a Python deep learning framework that lets you quickly put together neural network models with a minimal amount of code. It can be run on top of [Theano](http://deeplearning.net/software/theano/) or [Tensor Flow](https://www.tensorflow.org/) without you needing to know either of these underlying frameworks. It provides implementations of several of the layer architectures, objective functions, and optimization algorithms you need for building a model.

# ## <font color='#6629b2'>Dataset</font>
# 
# My research is on story generation, so I've selected a dataset of stories as the text to be modeled by the RNN. They come from the [ROCStories](http://cs.rochester.edu/nlp/rocstories/) dataset, which consists of thousands of five-sentence stories about everyday life events. Here the model will observe all five sentences in each story. Then we'll use the trained model to generate the final sentence in a set of stories not observed during training. The full dataset is available at the above link and just requires filling out a form to get access. Here, I'll use a sample of 100 stories.

# In[1]:

from __future__ import print_function #Python 2/3 compatibility for print statements


# I'll load the datasets using the [pandas library](https://pandas.pydata.org/), which is extremely useful for any task involving data storage and manipulation. This library puts a dataset into a readable table format, and makes it easy to retrieve specific columns and rows.

# In[2]:

'''Load the training dataset'''

import pandas

train_stories = pandas.read_csv('dataset/example_train_stories.csv', encoding='utf-8')#[:100]

train_stories[:10]


# ## <font color='#6629b2'>Preparing the data</font>
# 
# The model we'll create is a word-based language model, which means each input unit is a single word (some language models learn subword units like characters). 
# 
# 

# ###  <font color='#6629b2'>Tokenization</font>
# 
# The first pre-processing step is to tokenize each of the reviews into (lowercased) individual words, since the RNN will encode the reviews word by word. For this I'll use [spacy](https://spacy.io/), which is a fast and extremely user-friendly library that performs various language processing tasks. 

# In[3]:

'''Split texts into lists of words (tokens)'''

import spacy

encoder = spacy.load('en')

def text_to_tokens(text_seqs):
    token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
    return token_seqs

train_stories['Tokenized_Story'] = text_to_tokens(train_stories['Story'])
    
train_stories[['Story','Tokenized_Story']][:10]


# ###  <font color='#6629b2'>Lexicon</font>
# 
# Then we need to assemble a lexicon (aka vocabulary) of words that the model needs to know. Each tokenized word in the stories is added to the lexicon, and then each word is mapped to a numerical index that can be read by the model. Since large datasets may contain a huge number of unique words, it's common to filter all words occurring less than a certain number of times, and replace them with some generic &lt;UNK&gt; token. The min_freq parameter in the function below defines this threshold. In the example code, the min_freq parameter is set to 1, so the lexicon will contain all unique words in the training set. When assigning the indices, the number 1 will represent unknown words. The number 0 will represent "empty" word slots, which is explained below. Therefore "real" words will have indices of 2 or higher.

# In[4]:

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

    print(list(lexicon.items())[:20])
    
    return lexicon

lexicon = make_lexicon(token_seqs=train_stories['Tokenized_Story'], min_freq=1)
print("{} words in lexicon".format(len(lexicon)))

with open('example_model/lexicon.pkl', 'wb') as f: # Save the lexicon by pickling it
    pickle.dump(lexicon, f)


# Because the model will output tags as indices, we'll obviously need to map each tag number back to its corresponding string representation in order to later interpret the output. We'll reverse the tags lexicon to create a lookup table to get each tag from its index.

# In[5]:

'''Make a dictionary where the string representation of a lexicon item can be retrieved from its numerical index'''

def get_lexicon_lookup(lexicon):
    lexicon_lookup = {idx: lexicon_item for lexicon_item, idx in lexicon.items()}
    lexicon_lookup[0] = "" #map 0 padding to empty string
    print(list(lexicon_lookup.items())[:20])
    return lexicon_lookup

lexicon_lookup = get_lexicon_lookup(lexicon)


# ###  <font color='#6629b2'>From strings to numbers</font>
# 
# Once the lexicon is built, we can use it to transform each story from string tokens into a list of numerical indices.

# In[6]:

'''Convert each text from a list of tokens to a list of numbers (indices)'''

def tokens_to_idxs(token_seqs, lexicon):
    idx_seqs = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_seq]  
                                                                     for token_seq in token_seqs]
    return idx_seqs

train_stories['Story_Idxs'] = tokens_to_idxs(token_seqs=train_stories['Tokenized_Story'],
                                             lexicon=lexicon)
                                   
train_stories[['Tokenized_Story', 'Story_Idxs']][:10]


# ###  <font color='#6629b2'>Creating a matrix</font>
# 
# Finally, we need to put all the training stories into a single matrix, where each row is a story and each column is a word index in that story. This enables the model to process the stories in batches as opposed to one at a time, which significantly speeds up training. However, each story has a different number of words. So we create a padded matrix equal to the length on the longest story in the training set. For all stories with fewer words, we prepend the row with zeros representing an empty word position. Then we can actually tell Keras to ignore these zeros during training.

# In[7]:

'''create a padded matrix of stories'''

from keras.preprocessing.sequence import pad_sequences

def pad_idx_seqs(idx_seqs, max_seq_len):
    # Keras provides a convenient padding function; 
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len)
    return padded_idxs

max_seq_len = max([len(idx_seq) for idx_seq in train_stories['Story_Idxs']]) # Get length of longest sequence

train_padded_idxs = pad_sequences(train_stories['Story_Idxs'], maxlen=max_seq_len)
print(train_padded_idxs) #same example story as above

print("SHAPE:", train_padded_idxs.shape)


# ### <font color='#6629b2'>Defining the input and output</font>
# 
# In an RNN language model, the data is set up so that each word in the text is mapped to the word that follows it. In a given story, for each input word x[idx], the output label y[idx] is just x[idx+1]. In other words, the output sequences (y) matrix will be offset by one to the right. The example below displays this alignment with the string tokens for the first story in the dataset.

# In[9]:

pandas.DataFrame(list(zip(["-"] + train_stories['Tokenized_Story'].loc[0],
                          train_stories['Tokenized_Story'].loc[0])),
                 columns=['Input Word', 'Output Word'])


# To keep the padded matrices the same length, the input word matrix will also both be offset by one in the opposite direction. So the length of both the input and output matrices will be both reduced by one.

# In[10]:

print(pandas.DataFrame(list(zip(train_padded_idxs[0,:-1], train_padded_idxs[0, 1:])),
                columns=['Input Words', 'Output Words']))


# ##  <font color='#6629b2'>Building the model</font>
# 
# To assemble the model, we'll use Keras' [Functional API](https://keras.io/getting-started/functional-api-guide/), which is one of two ways to use Keras to assemble models (the alternative is the [Sequential API](https://keras.io/getting-started/sequential-model-guide/), which is a bit simpler but has more constraints). A model consists of a series of layers. As shown in the code below, we initialize instances for each layer. Each layer can be called with another layer as input, e.g. Embedding()(input_layer). A model instance is initialized with the Model() object, which defines the initial input and final output layers for that model. Before the model can be trained, the compile() function must be called with the loss function and optimization algorithm specified (see below).
# 
# ###  <font color='#6629b2'>Layers</font>
# 
# We'll build an RNN with five layers:
# 
# **1. Input**: The input layer takes in the matrix of word indices.
# 
# **2. Embedding**: An [embedding input layer](https://keras.io/layers/embeddings/) that converts word indices into distributed vector representations (embeddings). The mask_zero=True parameter indicates that values of 0 in the matrix (the padding) will be ignored by the model.
# 
# **3. GRU**: A [recurrent (GRU) hidden layer](https://keras.io/layers/recurrent/), the central component of the model. As it observes each word in the story, it integrates the word embedding representation with what it's observed so far to compute a representation (hidden state) of the review at that timepoint. There are a few architectures for this layer - I use the GRU variation, Keras also provides LSTM or just the simple vanilla recurrent layer (see the materials at the bottom for an explanation of the difference). By setting return_sequences=True for this layer, it will output the hidden states for every timepoint in the model, i.e. for every word in the story.
# 
# **4. GRU**: A second recurrent layer that takes the first as input and operates the same way, since adding more layers generally improves the model. Rather than returning the sequence of values for each word like the previous hidden layer, this layer will output just the last hidden state of the sequence (i.e. the hidden representation of the story after its last word is observed), since by default the return_sequences parameter is False.
# 
# **5. (Time Distributed) Dense**: A [dense output layer](https://keras.io/layers/core/#dense) that outputs a probability for each word in the lexicon, where each probability indicates the chance of that word being the next word in the sequence. The 'softmax' activation is what transforms the values of this layer into scores from 0 to 1 that can be treated as probabilities. The Dense layer produces the probability scores for one particular timepoint (word). By wrapping this in a TimeDistributed() layer, the model outputs a probability distribution for every timepoint in the sequence.
# 
# The term "layer" is just an abstraction, when really all these layers are just matrices. The "weights" that connect the layers are also matrices. The process of training a neural network is a series of matrix multiplications. The weight matrices are the values that are adjusted during training in order for the model to learn to predict the next word.
# 
# ###  <font color='#6629b2'>Parameters</font>
# 
# Our function for creating the model takes the following parameters:
# 
# **seq_input_len:** the length of the input and output matrices. This is equal to the length of the longest story in the training data. 
# 
# **n_input_nodes**: the number of unique words in the lexicon, plus one to account for the padding represented by 0 values. This indicates the number of rows in the embedding layer, where each row corresponds to a word. It is also the dimensionality of the probability vectors given as the model output.
# 
# **n_embedding_nodes**: the number of dimensions (units) in the embedding layer, which can be freely defined. Here, it is set to 300.
# 
# **n_hidden_nodes**: the number of dimensions in the hidden layers. Like the embedding layer, this can be freely chosen. Here, it is set to 500.
# 
# **stateful**: By default, the GRU hidden layer will reset its state (i.e. its values will be 0s) each time a new set of sequences is read into the model.  However, when stateful=True is given, this parameter indicates that the GRU hidden layer should "remember" its state until it is explicitly told to forget it. In other words, the values in this layer will be carried over between separate calls to the training function. This is useful when processing long sequences, so that the model can iterate through chunks of the sequences rather than loading the entire matrix at the same time, which is memory-intensive. I'll show below how this setting is also useful when the model is used for word prediction after training. During training, the model will observe all words in a story at once, so stateful will be set to False. At prediction time, it will be set to True.
# 
# **batch_size**: It is not always necessary to specify the batch size when setting up a Keras model. The fit() function will apply batch processing by default and the batch size can be given as a parameter. However, when a model is stateful, the batch size does need to be specified in the Input() layers. Here, for training, batch_size=None, so Keras will use its default batch size (which is 32). During prediction, the batch size will be set to 1.
# 
# ### <font color='#6629b2'>Procedure</font>
# 
# The output of the model is a sequence of vectors, each with the same number of dimensions as the number of unique words (n_input_nodes). Each vector contains the predicted probability of each possible word appearing in that position in the sequence. Like all neural networks, RNNs learn by updating the parameters (weights) to optimize an objective (loss) function applied to the output. For this model, the objective is to minimize the cross entropy (named as "sparse_categorical_crossentropy" in the code) between the predicted word probabilities and the probabilities observed from the words that appear in the training data, resulting in probabilities that more accurately predict when a particular word will appear. This is the general procedure used for all multi-label classification tasks. Updates to the weights of the model are performed using an optimization algorithm, such as Adam used here. The details of this process are extensive; see the resources at the bottom of the notebook if you want a deeper understanding. One huge benefit of Keras is that it implements many of these details for you. Not only does it already have implementations of the types of layer architectures, it also has many of the [loss functions](https://keras.io/losses/) and [optimization methods](https://keras.io/optimizers/) you need for training various models.

# In[11]:

'''Create the model'''

from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_model(seq_input_len, n_input_nodes, n_embedding_nodes, 
                 n_hidden_nodes, stateful=False, batch_size=None):
    
    # Layer 1
    input_layer = Input(batch_shape=(batch_size, seq_input_len), name='input_layer')

    # Layer 2
    embedding_layer = Embedding(input_dim=n_input_nodes, 
                                output_dim=n_embedding_nodes, 
                                mask_zero=True, name='embedding_layer')(input_layer) #mask_zero=True will ignore padding
    # Output shape = (batch_size, seq_input_len, n_embedding_nodes)

    #Layer 3
    gru_layer1 = GRU(n_hidden_nodes,
                     return_sequences=True, #return hidden state for each word, not just last one
                     stateful=stateful, name='hidden_layer1')(embedding_layer)
    # Output shape = (batch_size, seq_input_len, n_hidden_nodes)

    #Layer 4
    gru_layer2 = GRU(n_hidden_nodes,
                     return_sequences=True,
                     stateful=stateful, name='hidden_layer2')(gru_layer1)
    # Output shape = (batch_size, seq_input_len, n_hidden_nodes)

    #Layer 5
    output_layer = TimeDistributed(Dense(n_input_nodes,
                                         activation="softmax"), name='output_layer')(gru_layer2)
    # Output shape = (batch_size, seq_input_len, n_input_nodes)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    #Specify loss function and optimization algorithm, compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam')
    
    return model


# In[12]:

model = create_model(seq_input_len=train_padded_idxs.shape[-1] - 1, #substract 1 from matrix length because of offset 
                     n_input_nodes = len(lexicon) + 1, # Add 1 to account for 0 padding
                     n_embedding_nodes = 300,
                     n_hidden_nodes = 500)


# ###  <font color='#6629b2'>Training</font>
# 
# Now we're ready to train the model. We'll call the fit() function to train the model for 10 iterations through the dataset (epochs). Keras reports the cross-entropy loss after each epoch - if the model is learning correctly, it should progressively decrease.

# In[13]:

'''Train the model'''

# output matrix (y) has extra 3rd dimension added because sparse cross-entropy function requires one label per row
model.fit(x=train_padded_idxs[:,1:], y=train_padded_idxs[:, 1:, None], epochs=10)
model.save_weights('example_model/model_weights.h5') #Save model


# ## <font color='#6629b2'>Prediction Tasks</font>
# 
# Now that the model is trained, we can use it for prediction. I'll show two prediction tasks: computing a probability score for a story, and generating a new ending for a story. To demonstrate both of these, I'll load a saved model previously trained on all the 96,000 stories in the training set. As opposed to training where we processed multiple stories at the same time, it will be more straightforward to demonstrate prediction on a single story at a time, especially since prediction is fast relative to training. In Keras, you can duplicate a model by loading the parameters from a saved model into a new model. Here, this new model will have a batch size of 1. It will also process a story one word at a time (seq_input_len=1), using the stateful=True parameter to remember the story that has occurred up to that word. 

# In[20]:

'''Load test set and apply same processing used for training stories'''

test_stories = pandas.read_csv('dataset/example_test_stories.csv', encoding='utf-8')
test_stories['Tokenized_Story'] = text_to_tokens(test_stories['Story'])
test_stories['Story_Idxs'] = tokens_to_idxs(token_seqs=test_stories['Tokenized_Story'],
                                            lexicon=lexicon)


# In[15]:

'''Create a new test model, setting batch_size = 1, seq_input_len = 1, and stateful = True'''

# Load lexicon from the saved model 
with open('example_model/lexicon.pkl', 'rb') as f:
    lexicon = pickle.load(f)

predictor_model = create_model(seq_input_len=1,
                               n_input_nodes=len(lexicon) + 1,
                               n_embedding_nodes = 300,
                               n_hidden_nodes = 500,
                               stateful=True, 
                               batch_size = 1)

predictor_model.load_weights('pretrained_model/model_weights.h5') #Load weights from saved model


# ### <font color='#6629b2'>Computing story probabilities</font>

# Since the model outputs a probability distribution for each word in the story, indicating the probability of each possible next word in the story, we can use these values to get a single probability score for the story. To do this, we iterate through each word in a story, call the predict() function to get the full list of probabilites for the next word, and then extract the probability predicted for the actual next word in the story. We can average these probabilities across all words in the story to get a single value. The stateful=True parameter is what enables the model to remember the previous words in the story when predicting the probability of the next word. Becuase of this, the reset_states() function must be called at the end of reading the story in order to clear its memory for the next story.

# In[17]:

'''Compute the probability of a sequence according to the language model'''

import numpy

def get_probability(idx_seq):
    idx_seq = [0] + idx_seq #Prepend 0 so first call to predict() computes prob of first word from zero padding
    probs = []
    for word, next_word in zip(idx_seq[:-1], idx_seq[1:]):
       # Word is an integer, but the model expects an input array
       # with the shape (batch_size, seq_input_len), so prepend two dimensions
        p_next_word = predictor_model.predict(numpy.array(word)[None,None])[0,0] #Output shape= (lexicon_size + 1,)
        #Select predicted prob of the next word, which appears in the corresponding idx position of the probability vector
        p_next_word = p_next_word[next_word]
        probs.append(p_next_word)
    predictor_model.reset_states()
    return numpy.mean(probs) #return average probability of words in sequence

print("STORY:", test_stories['Story'].loc[0])
print("PROBABILITY:", get_probability(test_stories['Story_Idxs'].loc[0]))


# If we randomly shuffle all the words around in a story, for instance, the model should assign a much lower probability to the story:

# In[19]:

shuffled_word_positions = numpy.random.permutation(len(test_stories['Tokenized_Story'].loc[0]))
shuffled_token_story = [test_stories['Tokenized_Story'].loc[0][position] for position in shuffled_word_positions]
shuffled_idx_story = [test_stories['Story_Idxs'].loc[0][position] for position in shuffled_word_positions]
print("SHUFFLED STORY:", " ".join(shuffled_token_story))
print("PROBABILITY:", get_probability(shuffled_idx_story))


# ### <font color='#6629b2'>Generating sentences</font>
# 
# The language model can also be used to generate new text. Here, I'll give the same predictor model the first four sentences of a story in the test set and have it generate the fifth sentence. To do this, we "load" the first four sentences into the model. This can be done using predict() function. Because the model is stateful, predict() saves the representation of the story internally even though we don't need the output of this function when just reading the story. Once the final word in the fourth sentence has been read, then we can start using the resulting probability distribution to predict the first word in the fifth sentence. We can call numpy.random.choice() to randomly sample a word according to its probability. Now we again call predict() with this new word as input, which returns a probability distribution for the second word. Again, we sample from this distribution, append the newly sampled word to the previously generated word, and call predict() with this new word as input. We continue doing this until a word that ends with an end-of-sentence puncutation mark (".", "!", "?") has been selected. Just as before, reset_states() is called after the whole sentence has been generated. Then we can decode the generated ending into a string using the lexicon lookup dictionary. You can see that the generated endings are generally not as coherent and well-formed as the human-authored endings, but they do capture some components of the story and they are often more entertaining.

# In[22]:

'''Use the model to generate new endings for stories'''

def generate_ending(idx_seq):
    
    end_of_sent_tokens = [".", "!", "?"]
    generated_ending = []
    
    # First just read initial story, no output needed
    for word in idx_seq:
        p_next_word = predictor_model.predict(numpy.array(word)[None,None])[0,0]
        
    # Now start predicting new words
    while not generated_ending or lexicon_lookup[next_word] not in end_of_sent_tokens:
        #Randomly sample a word from the current probability distribution
        next_word = numpy.random.choice(a=p_next_word.shape[-1], p=p_next_word)
        # Append sampled word to generated ending
        generated_ending.append(next_word)
        # Get probabilities for next word by inputing sampled word
        p_next_word = predictor_model.predict(numpy.array(next_word)[None,None])[0,0]
    
    predictor_model.reset_states() #reset hidden state after generating ending
    
    return generated_ending

for _, test_story in test_stories[:20].iterrows():
    # Use spacy to segment the story into sentences, so we can seperate the ending sentence
    # Find out where in the story the ending starts (number of words from end of story)
    ending_story_idx = len(list(encoder(test_story['Story']).sents)[-1])
    print("INITIAL STORY:", " ".join(test_story['Tokenized_Story'][:-ending_story_idx]))
    print("GIVEN ENDING:", " ".join(test_story['Tokenized_Story'][-ending_story_idx:]))
    
    generated_ending = generate_ending(test_story['Story_Idxs'][:-ending_story_idx])
    generated_ending = " ".join([lexicon_lookup[word] if word in lexicon_lookup else ""
                                 for word in generated_ending]) #decode from numbers back into words
    print("GENERATED ENDING:", generated_ending, "\n")
    


# ### <font color='#6629b2'>Visualizing inner layers</font>
# 
# To help visualize the data representation inside the model, we can look at the output of each layer individually. Keras' Functional API lets you derive a new model with the layers from an existing model, so you can define the output to be a layer below the output layer in the original model. Calling predict() on this new model will produce the output of that layer for a given input. Of course, glancing at the numbers by themselves doesn't provide any interpretation of what the model has learned (although there are opportunities to [interpret these values](https://www.civisanalytics.com/blog/interpreting-visualizing-neural-networks-text-processing/)), but seeing them verifies the model is just a series of transformations from one matrix to another. The model stores its layers in the list model.layers, and you can retrieve specific layer by its position index in the model. Below is an example of the word embedding output for the first word in the first story of the test set. You can do this same thing to view any layer.

# In[23]:

'''Show the output of the word embedding layer'''

embedding_layer = Model(inputs=predictor_model.layers[0].input,
                        outputs=predictor_model.layers[1].output)
#Show output for first predicted tag in sequence (word input is first word, tag input is 0)
embedding_output = embedding_layer.predict(numpy.array(test_stories['Story_Idxs'][0][0])[None,None])
print("WORD EMBEDDINGS OUTPUT SHAPE:", embedding_output.shape)
print(embedding_output[0]) # Print embedding vectors for first word of first story


# ## <font color='#6629b2'>Conclusion</font>
# 
# There are a good number of tutorials on RNN language models, particularly applied to text genertion. This one shows how to leverage Keras with batch training when the length of the sequences is variable. There are many ways this language model can be made to be more sophisticated. Here's a few interesting papers from the NLP community that innovate this basic model for different generation tasks:
# 
# *Recipe generation:* [Globally Coherent Text Generation with Neural Checklist Models](https://homes.cs.washington.edu/~yejin/Papers/emnlp16_neuralchecklist.pdf). Chloé Kiddon, Luke Zettlemoyer, and Yejin Choi. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.
# 
# *Emotional text generation:* [Affect-LM: A Neural Language Model for Customizable Affective Text Generation](https://arxiv.org/pdf/1704.06851.pdf). Sayan Ghosh, Mathieu Chollet, Eugene Laksana, Louis-Philippe Morency, Stefan Scherer. Annual Meeting of the Association for Computational Linguistics (ACL), 2017.
# 
# *Poetry generation:* [Generating Topical Poetry](https://www.isi.edu/natural-language/mt/emnlp16-poetry.pdf). Marjan Ghazvininejad, Xing Shi, Yejin Choi, and Kevin Knight. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.
# 
# *Dialogue generation:* [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://www-etud.iro.umontreal.ca/~sordonia/pdf/naacl15.pdf). Alessandro Sordoni, Michel Galley, Michael Auli, Chris Brockett, Yangfeng Ji, Margaret Mitchell, Jian-Yun Nie1, Jianfeng Gao, Bill Dolan. North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2015.

# ## <font color='#6629b2'>More resources</font>
# 
# Yoav Goldberg's book [Neural Network Methods for Natural Language Processing](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) is a thorough introduction to neural networks for NLP tasks in general
# 
# If you'd like to learn more about what Keras is doing under the hood, the [Theano tutorials](http://deeplearning.net/tutorial/) are useful. There are two specifically on RNNs for NLP: [semantic parsing](http://deeplearning.net/tutorial/rnnslu.html#rnnslu) and [sentiment analysis](http://deeplearning.net/tutorial/lstm.html#lstm)
# 
# TensorFlow also has an RNN language model [tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/recurrent/index.html) using the Penn Treebank dataset
# 
# Andrej Karpathy's blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is very helpful for understanding the underlying details of the same language model I've demonstrated here. It also provides raw Python code with an implementation of the backpropagation algorithm.
# 
# Chris Olah provides a good [explanation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of how LSTM RNNs work (this explanation also applies to the GRU model used here)
# 
# Denny Britz's [tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) documents well both the technical details of RNNs and their implementation in Python.
# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



