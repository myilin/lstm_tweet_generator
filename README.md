# Tweet Generator

Character-level LSTM-based text generator, tuned to generate tweet-like chunks of text. When trained on the history of tweets written by one person, generated text hilariously resembles that person's writing style.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

You will also download the dataset of president Trump's funniest tweets to train the neural network on, and generate even funnier ones.

### Prerequisites

#### Nvidia GPU-powered computer
Highly recommended, otherwise training this network will take waaaay too long.

#### Python and Tensorflow for GPU
(alternatively, Theano or CNTK can be used as a backend for Keras, instead of Tensorflow).

For GPU owners, you'll also need CUDA Toolkit and CuDNN, as well as latest drivers for your GPU.

Follow setup instructions on [Keras.io](https://keras.io/#installation) and [Tensorflow.org](https://www.tensorflow.org/install) websites.

**Pay Attention** to a specific version of python, as well as CUDA Toolkit and CuDNN that are compatible with Tensorflow's pre-built binaries.

#### Python and Tensorflow for CPU

If all the GPU-related stuff doesn't apply to you, installing tensorflow should be as easy as running:
```
pip3 install --upgrade tensorflow
```
But still **Pay Attention** to a specific version of python compatible with Tensorflow's pre-built binaries.

#### Keras
Should be as simple as running:
```
pip install keras
```
For more details, refer to installation giude on [keras.io](https://keras.io/#installation).

#### Git client
Duh...
[git-scm.com/downloads](https://git-scm.com/downloads)

#### Virtualenv
Not a strict requirement.
However, if you're writing python code and are not using virtualenv, and just have been waiting for a divine sign to start doing so, *this is your divine sign*.
Follow instructions on [virtualenv.pypa.io](https://virtualenv.pypa.io/en/stable/)

#### matplotlib, numpy
For charts and stuff.
```
pip install matplotlib
pip install numpy
```

### Installing

Assuming that all prerequisites listed above are satisfied, here are your next steps:

- Clone lstm tweet generator repo.
```
git clone https://github.com/myilin/lstm_tweet_generator.git
```

- Clone Donald Trump's tweets repo.
```
git clone --depth 1 https://github.com/bpb27/trump_tweet_data_archive.git
```
'--depth 1' parameter specifies that we only want to fetch the latest revision of the repo, otherwise it will take waaay too long to clone it.

- Unzip them tweets right into the 'trump_tweet_data_archive' directory.

### Running

You're almost there!

Navigate to tweet generator directory and run the lstm text generator script:
```
cd tweet_generator
python .\lstm_text_generation.py
```

Yay! Now simply wait fot another century or two (if you're training it on a CPU), or about a day (GPU) for the network to train on about 20 epochs, and enjoy fresh faked tweets by your favorite author's artificial impresonator.

## Author

**Mykolai Ilin** - [github.com/myilin](https://github.com/myilin)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements

- Inspiration for the idea came from reading this awesome [blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Pieces of text generation functions taken from [Keras examples](https://github.com/keras-team/keras/tree/master/examples)
- Convenient access to tweets provided by [trumptwitterarchive.com](http://www.trumptwitterarchive.com/about)


