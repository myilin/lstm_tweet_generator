import numpy as np

from generation_helper import sample

class CharSequenceProvider:
    def initialize(self, full_text):
        # Getting the list of unique characters.
        self.chars = sorted(list(set(full_text)))
        print('total chars:' + str(len(self.chars)))

        # Mapping unique characters to indices and vice-versa.
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def getSequences(self, text, maxlen):
        """Cuts the text in semi-redundant sequences of maxlen characters."""

        print('corpus length:' + str(len(text)))

        step = 1
        sentences = []
        next_chars = []

        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        
        print('nb sequences:' + str(len(sentences)))

        print('Vectorization...')

        x = np.zeros((len(sentences), maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        return x, y

    def generateText(self, model, seed_sentence, generated_text_size, maxlen, temperature=1.0):
        """Generates text character-by-character, from the model in a given state, and a seed sentence."""

        generated = ''
        sentence = seed_sentence
        generated += sentence

        for i in range(generated_text_size):
            x_pred = np.zeros((1, maxlen, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        return generated