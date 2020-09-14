from emo_utils import *
from model_service import ModelService

if __name__ == '__main__':

    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')

    maxLen = len(max(X_train, key=len).split())

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

    model_service = ModelService()

    model = model_service.Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train_indices = model_service.sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)

    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

    X_test_indices = model_service.sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)

    C = 5
    y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    X_test_indices = model_service.sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    for i in range(len(X_test)):
        x = X_test_indices
        num = np.argmax(pred[i])
        if (num != Y_test[i]):
            print('Expected emoji:' + label_to_emoji(Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(
                num).strip())


