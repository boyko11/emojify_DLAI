import numpy as np

from emo_utils import convert_to_one_hot, softmax, predict

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding

np.random.seed(1)

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

    def sentences_to_indices(self, X, word_to_index, max_len):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """

        m = X.shape[0]  # number of training examples

        # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
        X_indices = np.zeros(shape=(X.shape[0], max_len))

        for i in range(m):  # loop over training examples

            # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
            sentence_words = X[i].lower().split()

            # Initialize j to 0
            j = 0

            # Loop over the words of sentence_words
            for w in sentence_words:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j += 1

        return X_indices

    # GRADED FUNCTION: pretrained_embedding_layer

    def pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """

        vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
        emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

        # Step 1
        # Initialize the embedding matrix as a numpy array of zeros.
        # See instructions above to choose the correct shape.
        emb_matrix = np.zeros(shape=(vocab_len, emb_dim))

        # Step 2
        # Set each row "idx" of the embedding matrix to be
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in word_to_index.items():
            emb_matrix[idx, :] = word_to_vec_map[word]

        # Step 3
        # Define Keras embedding layer with the correct input and output sizes
        # Make it non-trainable.
        embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

        # Step 4 (already done for you; please do not modify)
        # Build the embedding layer, it is required before setting the weights of the embedding layer.
        embedding_layer.build((None,))  # Do not modify the "None".  This line of code is complete as-is.

        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer

    # GRADED FUNCTION: Emojify_V2

    def Emojify_V2(self, input_shape, word_to_vec_map, word_to_index):
        """
        Function creating the Emojify-v2 model's graph.

        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        model -- a model instance in Keras
        """

        ### START CODE HERE ###
        # Define sentence_indices as the input of the graph.
        # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
        sentence_indices = Input(shape=input_shape, dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = self.pretrained_embedding_layer(word_to_vec_map, word_to_index)

        # Propagate sentence_indices through your embedding layer
        # (See additional hints in the instructions).
        embeddings = embedding_layer(sentence_indices)

        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        # The returned output should be a batch of sequences.
        X = LSTM(units=128, return_sequences=True)(embeddings)
        # Add dropout with a probability of 0.5
        X = Dropout(rate=0.5)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        # The returned output should be a single hidden state, not a batch of sequences.
        X = LSTM(units=128)(X)
        # Add dropout with a probability of 0.5
        X = Dropout(rate=0.5)(X)
        # Propagate X through a Dense layer with 5 units
        X = Dense(units=5)(X)
        # Add a softmax activation
        X = Activation('softmax')(X)

        # Create Model instance which converts sentence_indices into X.
        model = Model(inputs=sentence_indices, outputs=X)

        return model