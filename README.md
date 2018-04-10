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

Follow setup instructions on [Tensorflow.org](https://www.tensorflow.org/install).

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

- Unzip them tweets right into the ```trump_tweet_data_archive``` directory.

### Running

You're almost there!

Navigate to tweet generator directory and run the lstm text generator script:
```
cd tweet_generator
python .\lstm_text_generation.py
```

Wait for the script to finish running.
It may take about 10 minutes with default settings (on CPU).

Yay! Now enjoy fresh faked tweets by your favorite author's artificial impresonator, saved in 
```..\generated_data\latest_tweets.txt```

At this point it may look something like this:
```
the anan Trump Tump Tpun op ht bone Trone our Donee Trom cos fol tord
```
Not very impressive, but if you can see some text generated, the script works.
This is a result of training of a small lstm (2 layers, 32 neurons each), on a 1/1000 fraction of Mr. Trump's tweets. 

To improve quality of generated tweets, you need to train a much larger network, on a full dataset.

**Use of GPU is highly recommended.**

Training large network takes much longer.
On my NVidia GTX960 it takes about an hour per epoch, so 20 epochs will take ~20 hours.

Open file ```lstm_text_generation.py``` in your favorite editor, and change values of the following parameters:

```
num_layers = 3
num_neurons = 512
batch_size = 100
learning_rate = 0.0002
data_fraction = 1
generated_text_size = 2000
```

Don't forget to save changes you've made, and launch the script:
```
python .\lstm_text_generation.py
```

This time, results should look more like this:
```
"@JohnnyRack: @realDonaldTrump @DonaldJTrumpJr @realDonaldTrump @FoxNews I wonder why you will be the next POTUS. You are so well done.  Think Like a Champion
---
"@jaketapper: @realDonaldTrump @foxandfriends So right now that we will MAKE AMERICA GREAT AGAIN! https://t.co/kIP3r7ZBCc"
---
"@YirdTrump4USA: @realDonaldTrump YOU are saying I got us by the same golf course on safety &amp; Bring Hillary as they all be disgracedunt. - Donald Trump"
---
"@ronmeier123: @LeezaGibbons @Dropponsteeds @realDonaldTrump @foxandfriends this country needs to get your daughter, you\'re behind you and kill me to get this country in the most high!'
```
**Better, huh?**

****

## Author

**Mykolai Ilin** - [github.com/myilin](https://github.com/myilin)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements

- Inspiration for the idea came from reading this awesome [blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Pieces of text generation functions taken from [Keras examples](https://github.com/keras-team/keras/tree/master/examples)
- Convenient access to tweets provided by [trumptwitterarchive.com](http://www.trumptwitterarchive.com/about)