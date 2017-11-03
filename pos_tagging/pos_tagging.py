
# coding: utf-8

# # <font color='#6629b2'>Part-of-speech tagging with recurrent neural networks using Keras</font>
# ### https://github.com/roemmele/keras-rnn-demo/pos-tagging
# by Melissa Roemmele, 10/23/17, roemmele @ ict.usc.edu
# 
# ## <font color='#6629b2'>Overview</font>
# 
# I am going to show how to use the Keras library to build a recurrent neural network (RNN) model that labels part-of-speech (POS) tags for words in sentences. 
# 
# ### <font color='#6629b2'>Part-of-speech (POS) tagging</font>
# 
# A part-of-speech tag is the syntactic category associated with a particular word in a sentence, such as a noun, verb, preposition, determiner, adjective or adverb. Part-of-speech tagging is a fundamental task in natural language processing; see the [chapter in Juraksky & Martin's *Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/10.pdf) for more background about it. POS tagging is a common pre-processing step in many NLP pipeliens. For example, words with certain POS tags are more important than other words for capturing the content of a text (e.g. nouns and verbs carry more semantic meaning than grammatical words like prepositions and determiners), so models often take this into account when predicting the topic, sentiment, or some other categorical dimensions of a text. Start-of-the art models are quite successful, reaching near-perfect accuracy in the tags assigned to words. This tutorial will show you how to put together a simple tagger that uses a Recurrent Neural Network.
# 
# ### <font color='#6629b2'>Recurrent Neural Networks (RNNs)</font>
# 
# RNNs are a general framework for modeling sequence data and are particularly useful for natural language processing tasks. At a high level, RNN encode sequences via a set of parameters (weights) that are optimized to predict some output variable. The focus of this tutorial is on the code needed to assemble a model in Keras. For a more general introduction to RNNs, see the resources at the bottom. Here an RNN will be used to encode a sentence and assign of POS tag to each word. The model shown here is applicable to any dataset with a one-to-one mapping between the inputs and outputs. This involves any task where for each sequential unit (here, a word), there is some output unit (here, a POS tag) that should be assigned to that input unit.
# 
# ### <font color='#6629b2'>Keras</font>
# 
# [Keras](https://keras.io/) is a Python deep learning framework that lets you quickly put together neural network models with a minimal amount of code. It can be run on top of [Theano](http://deeplearning.net/software/theano/) or [Tensor Flow](https://www.tensorflow.org/) without you needing to know either of these underlying frameworks. It provides implementations of several of the layer architectures, objective functions, and optimization algorithms you need for building a model.

# ## <font color='#6629b2'>Dataset</font>
# 
# The [Brown Corpus](http://www.hit.uib.no/icame/brown/bcm.html) (download through NLTK [here](http://www.nltk.org/nltk_data/)) is a popular NLP resource that consists of 500 texts from a variety of sources, including news reports, academic essays, and fiction. Every word in the texts has been annotated with a POS tag. In this tutorial, each entry in the dataset is a single sentence. I split these sentences into training and testing sets of 51606 sentences and 5734 sentences, respectively.

# In[1]:

from __future__ import print_function #Python 2/3 compatibility for print statements


# I'll load the datasets using the [pandas library](https://pandas.pydata.org/), which is extremely useful for any task involving data storage and manipulation. This library puts a dataset into a readable table format, and makes it easy to retrieve specific columns and rows.

# In[2]:

'''Load the dataset'''

import pandas

train_sents = pandas.read_csv('dataset/train_brown_corpus.csv', encoding='utf-8')[:100] #For demo, load a sample


# In[3]:

#Get the tokens and tags into a readable list format

train_sents['Tokenized_Sentence'] = train_sents['Tokenized_Sentence'].apply(lambda sent: sent.lower().split("\t"))
train_sents['Tagged_Sentence'] = train_sents['Tagged_Sentence'].apply(lambda sent: sent.split("\t"))

train_sents[:10]


# ## <font color='#6629b2'>Preparing the data</font>
# 
# The sentences have already been tokenized, so both the words (tokens) in the sentence and the corresponding tags are represented as lists.
# 
# We need to assemble lexicons for both the words and tags. The purpose of the lexicon is to map each word/tag to a numerical index that can be read by the model. For the words lexicon, since large datasets may contain a huge number of unique words, it's common to filter all words occurring less than a certain number of times and replace them with some generic &lt;UNK&gt; token. The min_freq parameter in the function below defines this threshold. For the tags, we'll include all of them in the model since these are the output classes we are trying to predict. There are only 11 tags in this dataset.

# ###  <font color='#6629b2'>Lexicon</font>

# In[4]:

'''Create a lexicon for the words in the sentences as well as the tags'''

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
    # Indices start at 1. 0 is reserved for padding, and 1 is reserved for unknown words.
    lexicon = {token:idx + 2 for idx,token in enumerate(lexicon)}
    lexicon[u'<UNK>'] = 1 # Unknown words are those that occur fewer than min_freq times
    lexicon_size = len(lexicon)

    print(list(lexicon.items())[:20])
    
    return lexicon

words_lexicon = make_lexicon(train_sents['Tokenized_Sentence'])
with open('pretrained_model/words_lexicon.pkl', 'wb') as f: #save the tags lexicon by pickling it
    pickle.dump(words_lexicon, f)
    
tags_lexicon = make_lexicon(train_sents['Tagged_Sentence'])
with open('pretrained_model/tags_lexicon.pkl', 'wb') as f: #save the words lexicon by pickling it
    pickle.dump(tags_lexicon, f)


# Because the model will output tags as indices, we'll obviously need to map each tag number back to its corresponding string representation in order to later interpret the output. We'll reverse the tags lexicon to create a lookup table to get each tag from its index.

# In[5]:

'''Make a dictionary where the string representation of a lexicon item can be retrieved from its numerical index'''

def get_lexicon_lookup(lexicon):
    lexicon_lookup = {idx: lexicon_item for lexicon_item, idx in lexicon.items()}
    print(list(lexicon_lookup.items())[:20])
    return lexicon_lookup

tags_lexicon_lookup = get_lexicon_lookup(tags_lexicon)


# ###  <font color='#6629b2'>From strings to numbers</font>

# We use the lexicons to transform the word and tag sequences into lists of numerical indices.

# In[6]:

def tokens_to_idxs(token_seqs, lexicon):
    idx_seqs = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_seq]  
                                                                     for token_seq in token_seqs]
    return idx_seqs

train_sents['Sentence_Idxs'] = tokens_to_idxs(train_sents['Tokenized_Sentence'], words_lexicon)

train_sents['Tag_Idxs'] = tokens_to_idxs(train_sents['Tagged_Sentence'], tags_lexicon)

train_sents[['Tokenized_Sentence', 'Sentence_Idxs', 'Tagged_Sentence', 'Tag_Idxs']][:10]


# ###  <font color='#6629b2'>Numerical lists to matrices</font>
# 
# Finally, we need to put the input sequences into matrices for training. There will be separate matrices for the word and tag sequences, where each row is a sentence and each column is a word (or tag) index in that sentence. This matrix format is necessary for the model to process the sentences in batches as opposed to one at a time, which significantly speeds up training. However, each sentence has a different number of words, so we create a padded matrix equal to the length on the longest sentence in the training set. For all sentences with fewer words, we prepend the row with zeros representing an empty word (and tag) position. We can specify to Keras to ignore these zeros during training.

# In[7]:

from keras.preprocessing.sequence import pad_sequences

def pad_idx_seqs(idx_seqs, max_seq_len):
    # Keras provides a convenient padding function; 
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len)
    return padded_idxs

max_seq_len = max([len(idx_seq) for idx_seq in train_sents['Sentence_Idxs']]) # Get length of longest sequence
train_padded_words = pad_idx_seqs(train_sents['Sentence_Idxs'], 
                                  max_seq_len + 1) #Add one to max length for offsetting sequence by 1
train_padded_tags = pad_idx_seqs(train_sents['Tag_Idxs'],
                                 max_seq_len + 1)  #Add one to max length for offsetting sequence by 1

print("WORDS:", train_padded_words)
print("SHAPE:", train_padded_words.shape, "\n")

print("TAGS:", train_padded_tags)
print("SHAPE:", train_padded_tags.shape, "\n")


# ### <font color='#6629b2'>Defining the input and output</font>
# 
# In this approach, for each word in a sentence, we predict the tag for that word based on two types of input: 1. all the words in the sentence up to that point, including that current word, and 2. all the previous tags in the sentence. So for a given position in the sentence *idx*, the input is train_padded_words[idx] and train_padded_tags[idx-1], and the output is train_padded_tags[idx]. In other words, the input tags matrix will be offset by one. The example below shows this alignment for the first sentence in the dataset.
# 

# In[8]:

import numpy

pandas.DataFrame(list(zip(train_sents['Tokenized_Sentence'].loc[0],
                          ["-"] + train_sents['Tagged_Sentence'].loc[0],
                          train_sents['Tagged_Sentence'].loc[0])),
                 columns=['Input Word', 'Input Tag', 'Output Tag'])


# As described above, the input tag sequences are offset by one so that each tag is predicted based on the tag sequence up to that point. To keep the padded matrices the same length, the input word matrix and output tag matrix will also both be offset by one in the opposite direction. Thus the length of all matrices will be reduced by one.

# In[9]:

print(pandas.DataFrame(list(zip(train_padded_words[0,1:], train_padded_tags[0,:-1], train_padded_tags[0, 1:])),
                columns=['Input Words', 'Input Tags', 'Output Tags']))


# ##  <font color='#6629b2'>Building the model</font>
# 
# ### <font color='#6629b2'>Functional API</font>
# 
# To set up the model, we'll use Keras [Functional API](https://keras.io/getting-started/functional-api-guide/), which is one of two ways to assemble models in Keras (the alternative is the [Sequential API](https://keras.io/getting-started/sequential-model-guide/), which is a bit simpler but has more constraints). For the POS tagger model, new tags will be predicted from the combination of two input sequences, the words in the sentence and the corresponding tags in the sentence. The Functional API is specifically useful when a model has multiple inputs and/or outputs. Compared to the Sequential model where the order in which the layers are added indicates which layers are connected, in the Functional API, the input to a particular layer must be specified as a parameter, e.g. Embedding()(word_input). This is what enables the model to have multiple inputs and outputs. A model is defined with the Model() function, and there the list of inputs and outputs is explicitly given. 
# 
# ### <font color='#6629b2'>Layers</font>
# 
# We'll build an RNN with the following layers, numbered according to the level on which they are stacked:
# 
# **1. Input (words)**: This input layer takes in a sequence of word indices.
# 
# **1. Input (tags)**: This is the other input layer alongside the first, and it takes in a sequence of tag indices. It is on the same level as the word input layer, so both input sequences are read in parallel by the model.
# 
# **2. Embedding (words)**: There are two embedding layers, one for the words and a different one for the tags. Both of them function the same way: they convert the indices into distributed vector representations (embeddings). The mask_zero=True parameter indicates that values of 0 in the matrix (the padding) will be ignored by the model.
# 
# **2. Embedding (tags)**: Same as the word embedding layer, but for the tags.
# 
# **3. Concatenate**: This layer merges each embedded word sequence and corresponding embedded tag sequence into a single sequence. This means that for a given word and the tag for that word, their vectors will be concatenated into a single vector.
# 
# **4. GRU**: The recurrent (GRU) hidden layer reads the merged embedded sequence and computes a representation (hidden state) of the sequence. The result is a new vector for each word/tag in the sequence. There are a few architectures for this layer - I use the GRU variation, Keras also provides LSTM or just the simple vanilla recurrent layer. By specifying return_sequences=True in the below function, this layer will output the entire sequence of vectors (hidden states) for the sequence, rather than just the most recent hidden state that is returned by default.
# 
# **5. (Time Distributed) Dense**: An output layer that produces a probability distribution for each possible tag for each word in the sequence. The 'softmax' activation is what transforms the values of this layer into scores from 0 to 1 that can be treated as probabilities. The Dense layer produces the probability scores for one particular timepoint (word). By wrapping this in a TimeDistributed() layer, the model outputs a probability distribution for every timepoint in the sequence. 
# 
# Each layer is connected to the layer above it via a set of weights, which are the parameters that are adjusted during training in order for the model to learn to predict tags. 
# 
# ### <font color='#6629b2'>Parameters</font>
# 
# Our function for creating the model takes the following parameters:
# 
# **seq_input_length**: the length of the padded matrices for the word and tag sentence inputs, which will be the same since there is a one-to-one mapping between tags. This is equal to the length of the longest sentence in the training data. 
# 
# **n_word_input_nodes**: the number of unique words in the lexicon, plus one to account for matrix padding represented by 0 values. This indicates the number of rows in the word embedding layer, where each row corresponds to a word.
# 
# **n_tag_input_nodes**: the number of unique tags in the dataset, plus one to account for padding. This indicates the number of rows in the tag embedding layer, where each row corresponds to a tag.
# 
# **n_word_embedding_nodes**: the number of dimensions in the word embedding layer, which can be freely defined. Here, it is set to 300.
# 
# **n_tag_embedding_nodes**: the number of dimensions in the tag embedding layer, which can be freely defined. Here, it is set to 100.
# 
# **n_hidden_nodes**: the number of dimensions in the hidden layer. Like the embedding layers, this can be freely chosen. Here, it is set to 500.
# 
# **stateful**: By default, the GRU hidden layer will reset its state (i.e. its values will be 0s) each time a new set of sequences is read into the model.  However, when stateful=True is given, this parameter indicates that the GRU hidden layer should "remember" its state until it is explicitly told to forget it. In other words, the values in this layer will be carried over between separate calls to the training function. This is useful when processing long sequences, so that the model can iterate through chunks of the sequences rather than loading the entire matrix at the same time, which is memory-intensive. I'll show below how this setting is also useful when tagging new sequences. Here, because the training sequences only consist of one sentence, stateful will be set to False during training. At prediction time, it will be set to True.
# 
# **batch_size**: It is not always necessary to specify the batch size when setting up a Keras model. The fit() function will apply batch processing by default and the batch size can be given as a parameter. However, when a model is stateful, the batch size does need to be specified in the Input() layers. Here, for training, batch_size=None, so Keras will use its default batch size (which is 32). During prediction, the batch size will be set to 1.
# 
# ### <font color='#6629b2'>Procedure</font>
# 
# The output of the model is a sequence of vectors, each with the same number of dimensions as the number of unique tags (n_tag_input_nodes). Each vector contains the predicted probability of each possible tag for the corresponding word in that position in the sequence. Like all neural networks, RNNs learn by updating the parameters (weights) to optimize an objective (loss) function applied to the output. For this model, the objective is to minimize the cross entropy (named as the "sparse_categorical_crossentropy" in the code) between the predicted tag probabilities and the probabilities observed from the words in training data, resulting in probabilities that more accurately predict when a particular tag will appear. This is the general procedure used for all multi-label classification tasks. Updates to the weights of the model are performed using an optimization algorithm, such as Adam used here. The details of this process are extensive; see the resources at the bottom of the notebook if you want a deeper understanding. One huge benefit of Keras is that it implements many of these details for you. Not only does it already have implementations of the types of layer architectures, it also has many of the [loss functions](https://keras.io/losses/) and [optimization methods](https://keras.io/optimizers/) you need for training various models.
# 

# In[10]:

'''Create the model'''

from keras.models import Model
from keras.layers import Input, Concatenate, TimeDistributed, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_model(seq_input_len, n_word_input_nodes, n_tag_input_nodes, n_word_embedding_nodes,
                 n_tag_embedding_nodes, n_hidden_nodes, stateful=False, batch_size=None):
    
    #Layers 1
    word_input = Input(batch_shape=(batch_size, seq_input_len), name='word_input_layer')
    tag_input = Input(batch_shape=(batch_size, seq_input_len), name='tag_input_layer')

    #Layer 2
    word_embeddings = Embedding(input_dim=n_word_input_nodes,
                                output_dim=n_word_embedding_nodes, 
                                mask_zero=True, name='word_embedding_layer')(word_input)
    #Output shape = (batch_size, seq_input_len, n_word_embedding_nodes)
    tag_embeddings = Embedding(input_dim=n_tag_input_nodes,
                               output_dim=n_tag_embedding_nodes,
                               mask_zero=True, name='tag_embedding_layer')(tag_input) 
    #Output shape = (batch_size, seq_input_len, n_tag_embedding_nodes)
    
    #Layer 3
    merged_embeddings = Concatenate(axis=-1, name='concat_embedding_layer')([word_embeddings, tag_embeddings])
    #Output shape =  (batch_size, seq_input_len, n_word_embedding_nodes + n_tag_embedding_nodes)
    
    #Layer 4
    hidden_layer = GRU(units=n_hidden_nodes, return_sequences=True, 
                       stateful=stateful, name='hidden_layer')(merged_embeddings)
    #Output shape = (batch_size, seq_input_len, n_hidden_nodes)
    
    #Layer 5
    output_layer = TimeDistributed(Dense(units=n_tag_input_nodes, 
                                         activation='softmax'), name='output_layer')(hidden_layer)
    # Output shape = (batch_size, seq_input_len, n_tag_input_nodes)
    
    #Specify which layers are input and output, compile model with loss and optimization functions
    model = Model(inputs=[word_input, tag_input], outputs=output_layer)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam')
    
    return model


# In[11]:

model = create_model(seq_input_len=train_padded_words.shape[-1] - 1, #substract 1 from matrix length because of offset
                     n_word_input_nodes=len(words_lexicon) + 1, #Add one for 0 padding
                     n_tag_input_nodes=len(tags_lexicon) + 1, #Add one for 0 padding
                     n_word_embedding_nodes=300,
                     n_tag_embedding_nodes=100,
                     n_hidden_nodes=500)


# ### <font color='#6629b2'>Training</font>
# 
# Now we're ready to train the model. We'll call the fit() function to train the model for 10 iterations through the dataset (epochs). Keras reports to cross-entropy loss after each epoch, which should continue to decrease if the model is learning correctly.

# In[12]:

'''Train the model'''

# output matrix (y) has extra 3rd dimension added because sparse cross-entropy function requires one label per row
model.fit(x=[train_padded_words[:,1:], train_padded_tags[:,:-1]], y=train_padded_tags[:, 1:, None], epochs=10)
model.save_weights('pretrained_model/model_weights.h5') #Save model


# ## <font color='#6629b2'>Tagging new sentences</font>
# 
# Now that the model is trained, it can be used to predict tags in new sentences in the test set. As opposed to training where we processed multiple sentences at the same time, it will be more straightforward to demonstrate tagging on a single sentence at a time. In Keras, you can duplicate a model by loading the parameters from a saved model into a new model. Here, this new model will have a batch size of 1. It will also process a sentence one word/tag at a time (seq_input_len=1) and predict the next tag, using the stateful=True parameter to remember its previous predictions within that sentence. The other parameters of this prediction model are exactly the same as the trained model. To demonstrate prediction performance, I'll load the weights from a saved model previously trained on the full training set of 51606 sentences. 
# 

# In[13]:

'''Load the test set and apply same processing steps performed above for training set'''

test_sents = pandas.read_csv('dataset/test_brown_corpus.csv', encoding='utf-8')[:100] #Load sample of 100 sentences
test_sents['Tokenized_Sentence'] = test_sents['Tokenized_Sentence'].apply(lambda sent: sent.lower().split("\t"))
test_sents['Tagged_Sentence'] = test_sents['Tagged_Sentence'].apply(lambda sent: sent.split("\t"))
test_sents['Sentence_Idxs'] = tokens_to_idxs(test_sents['Tokenized_Sentence'], words_lexicon)
test_sents['Tag_Idxs'] = tokens_to_idxs(test_sents['Tagged_Sentence'], tags_lexicon)


# In[14]:

'''Create predictor model with weights from saved model, with batch_size = 1, seq_input_len = 1 and stateful=True'''

# Load word and tag lexicons from the saved model 
with open('pretrained_model/words_lexicon.pkl', 'rb') as f:
    words_lexicon = pickle.load(f)
    
# Load word and tag lexicons from the saved model 
with open('pretrained_model/tags_lexicon.pkl', 'rb') as f:
    tags_lexicon = pickle.load(f)

predictor_model = create_model(seq_input_len=1,
                               n_word_input_nodes=len(words_lexicon) + 1,
                               n_tag_input_nodes=len(tags_lexicon) + 1,
                               n_word_embedding_nodes=300,
                               n_tag_embedding_nodes=100,
                               n_hidden_nodes=500,
                               stateful=True,
                               batch_size=1)

#Transfer the weights from the trained model
predictor_model.load_weights('pretrained_model/model_weights.h5')


# We'll iterate through the sentences in the test set and tag each of them. For each sentence, we start with an empty list for the predicted tags. For the first word in the sentence, there is no previous tag, so the model reads that word and the empty tag 0 (the padding value). The predict() function returns a probability distribution over the tags, and we pick the tag with the highest probability as the one to assign that word. This tag is appended to our list of predicted tags, and we continue to the next word in the sentence. Because the model is stateful, we can simply provide the current word and most recent tag as input to the predict() function, since it has saved the state of the model after the previous prediction. After the entire sentence has been tagged, we call reset_states() to clear the values for this sentence so we can process a new sentence. The tag indices are mapped back to their string forms, which we show in the sample below.
# 

# In[15]:

import numpy

if __name__ == '__main__':
    pred_tags = []
    for _, sent in test_sents.iterrows():
        tok_sent = sent['Tokenized_Sentence']
        sent_idxs = sent['Sentence_Idxs']
        sent_gold_tags = sent['Tagged_Sentence']
        sent_pred_tags = []
        prev_tag = 0  #initialize predicted tag sequence with padding
        for cur_word in sent_idxs:
            # cur_word and prev_tag are just integers, but the model expects an input array
            # with the shape (batch_size, seq_input_len), so prepend two dimensions to these values
            p_next_tag = predictor_model.predict(x=[numpy.array(cur_word)[None, None],
                                                        numpy.array(prev_tag)[None, None]])[0]
            prev_tag = numpy.argmax(p_next_tag, axis=-1)[0]
            sent_pred_tags.append(prev_tag)
        predictor_model.reset_states()
        
        #Map tags back to string labels
        sent_pred_tags = [tags_lexicon_lookup[tag] for tag in sent_pred_tags]
        pred_tags.append(sent_pred_tags) #filter padding
        
        print("SENTENCE:\t{}".format("\t".join(tok_sent)))
        print("PREDICTED:\t{}".format("\t".join(sent_pred_tags)))
        print("GOLD:\t\t{}".format("\t".join(sent_gold_tags)), "\n\n")
    
    test_sents['Predicted_Tagged_Sentence'] = pred_tags


# ### <font color='#6629b2'>Visualizing inner layers</font>
# 
# To help visualize the data representation inside the model, we can look at the output of each layer individually. Keras' Functional API lets you derive a new model with the layers from an existing model, so you can define the output to be a layer below the output layer in the original model. Calling predict() on this new model will produce the output of that layer for a given input. Of course, glancing at the numbers by themselves doesn't provide any interpretation of what the model has learned (although there are opportunities to [interpret these values](https://www.civisanalytics.com/blog/interpreting-visualizing-neural-networks-text-processing/)), but seeing them verifies the model is just a series of transformations from one matrix to another. The get_layer() function lets you retrieve any layer by the name that was assigned to it when creating the model. Below is an example of the output for the tag embedding layer for the first word in the first sentence of the test set. You can do this same thing to view any layer.

# In[16]:

'''Show the output of the tag embedding layer'''

tag_embedding_layer = Model(inputs=[predictor_model.get_layer('word_input_layer').input,
                                    predictor_model.get_layer('tag_input_layer').input], 
                            outputs=predictor_model.get_layer('tag_embedding_layer').output)
#Show output for first predicted tag in sequence (word input is first word, tag input is 0)
tag_embedding_output = tag_embedding_layer.predict([numpy.array(test_sents['Sentence_Idxs'][0][0])[None,None], 
                                                    numpy.array(0)[None,None]])
print("TAG EMBEDDINGS OUTPUT SHAPE:", tag_embedding_output.shape)
print(tag_embedding_output[0]) # Print embedding vectors for first review in test set


# ### <font color='#6629b2'>Evaluation</font>
# 
# We can evaluate our model with some of the standard metrics for classification: *precision*, *recall*, and *F1 score*. In the context of this task, precision is the proportion of the predicted tags for a particular class that were correct predictions (i.e. of all the words that were assigned a NOUN tag by the tagger, what percentage of these were actually nouns according to the test set?). Recall is the proportion of correct tags for a particular class that the tagger also predicted correctly (i.e. of all the words in the test set that should have been assigned a NOUN tag, what percentage of these were actually tagged as a NOUN?). F1 score is a weighted average of precision and recall. The scikit-learn package has several of these [evaluation metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) available.

# In[17]:

'''Evalute the model by precision, recall, and F1'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
    all_gold_tags = [tag for sent_tags in test_sents['Tagged_Sentence'] for tag in sent_tags]
    all_pred_tags = [tag for sent_tags in test_sents['Predicted_Tagged_Sentence'] for tag in sent_tags]
    accuracy = accuracy_score(y_true=all_gold_tags, y_pred=all_pred_tags)
    precision = precision_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')
    recall = recall_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')
    f1 = f1_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')

    print("ACCURACY: {:.3f}".format(accuracy))
    print("PRECISION: {:.3f}".format(precision))
    print("RECALL: {:.3f}".format(recall))
    print("F1: {:.3f}".format(f1))


# ## <font color='#6629b2'>Conclusion</font>
# 
# Even though this model can accuractely predict many POS tags, state-of-the-art taggers use more sophisticated techniques. For example, where here we predicted a tag just based on the preceding words and tags, [bidirectional layers](https://keras.io/layers/wrappers/#bidirectional) also model the sequence that appears after the given word to additionally inform the prediction. POS tagging can be seen as a shallow version of syntactic parsing, which is a more difficult NLP problem. Where POS tagging outputs a flat sequence with a one-to-one mapping between words and tags, syntatic parsing produces a hierarchical structure where categories consist of multiple-word phrases and phrase categories are embedded inside other phrases. Check out the [chapter from Jurafsky & Martin's book](https://web.stanford.edu/~jurafsky/slp3/14.pdf) if you're interested in learning more about these deeper models of linguistic structure.

# ## <font color='#6629b2'>More resources</font>
# 
# Yoav Goldberg's book [Neural Network Methods for Natural Language Processing](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) is a thorough introduction to neural networks for NLP tasks in general
# 
# If you'd like to learn more about what Keras is doing under the hood, the [Theano tutorials](http://deeplearning.net/tutorial/) are useful. There is one specifically on [semantic parsing](http://deeplearning.net/tutorial/rnnslu.html#rnnslu), which is related to the POS tagging task.
# 
# TensorFlow also has an RNN language model [tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/recurrent/index.html) using the Penn Treebank dataset
# 
# Andrej Karpathy's blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is very helpful for understanding the underlying details of the same language model I've demonstrated here. It also provides raw Python code with an implementation of the backpropagation algorithm.
# 
# Chris Olah provides a good [explanation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of how LSTM RNNs work (this explanation also applies to the GRU model used here)
# 
# Denny Britz's [tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) documents well both the technical details of RNNs and their implementation in Python.
# 
