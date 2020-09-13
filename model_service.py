import numpy as np
from emo_utils import convert_to_one_hot, softmax, predict

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

    # GRADED FUNCTION: model

    def model(self, X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
        """
        Model to train word vector representations in numpy.

        Arguments:
        X -- input data, numpy array of sentences as strings, of shape (m, 1)
        Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        learning_rate -- learning_rate for the stochastic gradient descent algorithm
        num_iterations -- number of iterations

        Returns:
        pred -- vector of predictions, numpy-array of shape (m, 1)
        W -- weight matrix of the softmax layer, of shape (n_y, n_h)
        b -- bias of the softmax layer, of shape (n_y,)
        """

        np.random.seed(1)

        # Define number of training examples
        m = Y.shape[0]  # number of training examples
        n_y = 5  # number of classes
        n_h = 50  # dimensions of the GloVe vectors

        # Initialize parameters using Xavier initialization
        W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
        b = np.zeros((n_y,))

        # Convert Y to Y_onehot with n_y classes
        Y_oh = convert_to_one_hot(Y, C=n_y)

        # Optimization loop
        for t in range(num_iterations):  # Loop over the number of iterations
            for i in range(m):  # Loop over the training examples

                # Average the word vectors of the words from the i'th training example
                avg = self.sentence_to_avg(X[i], word_to_vec_map)

                # Forward propagate the avg through the softmax layer
                z = np.dot(W, avg) + b
                a = softmax(z)

                # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
                cost = -np.sum(Y_oh[i] * np.log(a))

                # Compute gradients
                dz = a - Y_oh[i]
                dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
                db = dz

                # Update parameters with Stochastic Gradient Descent
                W = W - learning_rate * dW
                b = b - learning_rate * db

            if t % 100 == 0:
                print("Epoch: " + str(t) + " --- cost = " + str(cost))
                pred = predict(X, Y, W, b, word_to_vec_map)  # predict is defined in emo_utils.py

        return pred, W, b