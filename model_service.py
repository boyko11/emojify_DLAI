import numpy as np

class ModelService:

    def __init__(self):
        pass

    # GRADED FUNCTION: sentence_to_avg

    def sentence_to_avg(self, sentence, word_to_vec_map):
        """
        Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
        and averages its value into a single vector encoding the meaning of the sentence.

        Arguments:
        sentence -- string, one training example from X
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

        Returns:
        avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
        """

        # Step 1: Split sentence into list of lower case words (≈ 1 line)
        words = sentence.lower().split()

        # Initialize the average word vector, should have the same shape as your word vectors.
        total = np.zeros(shape=word_to_vec_map[words[0]].shape)

        # Step 2: average the word vectors. You can loop over the words in the list "words".
        for w in words:
            total += word_to_vec_map[w]
        avg = total / float(len(words))

        return avg