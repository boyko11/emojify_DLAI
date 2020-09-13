import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt


if __name__ == '__main__':

    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')

    maxLen = len(max(X_train, key=len).split())

    for idx in range(10):
        print(X_train[idx], label_to_emoji(Y_train[idx]))

    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)

    idx = 50
    print(f"Sentence '{X_train[50]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
    print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

    word = "cucumber"
    idx = 289846
    print("the index of", word, "in the vocabulary is", word_to_index[word])
    print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])


