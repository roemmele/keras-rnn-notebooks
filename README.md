# keras-rnn-demo
A brief tutorial that uses Keras to build a basic Recurrent Neural Network Language Model and uses it to generate story text.

To run the code in the notebook, you'll need [Numpy](http://www.numpy.org/), [Theano](http://deeplearning.net/software/theano/), and [Keras](https://keras.io/).

The dataset used in this tutorial is the [ROCStories corpus](http://cs.rochester.edu/nlp/rocstories/). You'll just need to complete a form to get access to it. Just replace the filepath 'ROCStories_winter2017.csv' in the notebook with the filepath to the downloaded file.

By default the tokenizer and trained model will be saved to 'tokenizer.pkl' and 'rnn.h5', respectively. I've included a model, rnn_96000.h5, that was trained on 96,0000 stories in the ROCStories corpus. The accompanying tokenizer (which contains the lexicon) is tokenizer_96000.pkl. If you just want to run the trained model to generate text, you can skip to the "Generating sentences" section in the notebook. 
