import unittest
from emo_utils import read_glove_vecs, read_csv
from model_service import ModelService
import numpy as np


class ModelServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.words, cls.index_to_words, cls.word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
        cls.model_service = ModelService()

    def test_sentence_to_avg(self):

        avg = np.round(self.model_service.sentence_to_avg("Morrocan couscous is my favorite dish", self.word_to_vec_map), 3)
        expected_avg = np.round(np.array(
                                [-0.008005, 0.56370833, -0.50427333, 0.258865, 0.55131103, 0.03104983,
                                 -0.21013718, 0.16893933, -0.09590267, 0.141784, -0.15708967, 0.18525867,
                                 0.6495785, 0.38371117, 0.21102167, 0.11301667, 0.02613967, 0.26037767,
                                 0.05820667, -0.01578167, -0.12078833, -0.02471267, 0.4128455, 0.5152061,
                                 0.38756167, -0.898661, -0.535145, 0.33501167, 0.68806933, -0.2156265,
                                 1.797155, 0.10476933, -0.36775333, 0.750785, 0.10282583, 0.348925,
                                 -0.27262833, 0.66768, -0.10706167, -0.283635, 0.59580117, 0.28747333,
                                 -0.3366635, 0.23393817, 0.34349183, 0.178405, 0.1166155, -0.076433,
                                 0.1445417, 0.09808667]
        ), 3)

        self.assertListEqual(expected_avg.tolist(), avg.tolist())

    def test_model(self):

        X_train, Y_train = read_csv('data/train_emoji.csv')
        print(X_train.shape)
        print(Y_train.shape)
        print(np.eye(5)[Y_train.reshape(-1)].shape)
        print(X_train[0])
        print(type(X_train))
        Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
        print(Y.shape)

        X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
                        'Lets go party and drinks', 'Congrats on the new job', 'Congratulations',
                        'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
                        'You totally deserve this prize', 'Let us go play football',
                        'Are you down for football this afternoon', 'Work hard play harder',
                        'It is suprising how people can be dumb sometimes',
                        'I am very disappointed', 'It is the best day in my life',
                        'I think I will end up alone', 'My life is so boring', 'Good job',
                        'Great so awesome'])

        print(X.shape)
        print(np.eye(5)[Y_train.reshape(-1)].shape)
        print(type(X_train))

        pred, W, b = self.model_service.model(X_train, Y_train, self.word_to_vec_map)

        expected_pred = np.array([[ 3.],
                                     [ 2.],
                                     [ 3.],
                                     [ 0.],
                                     [ 4.],
                                     [ 0.],
                                     [ 3.],
                                     [ 2.],
                                     [ 3.],
                                     [ 1.],
                                     [ 3.],
                                     [ 3.],
                                     [ 1.],
                                     [ 3.],
                                     [ 2.],
                                     [ 3.],
                                     [ 2.],
                                     [ 3.],
                                     [ 1.],
                                     [ 2.],
                                     [ 3.],
                                     [ 0.],
                                     [ 2.],
                                     [ 2.],
                                     [ 2.],
                                     [ 1.],
                                     [ 4.],
                                     [ 3.],
                                     [ 3.],
                                     [ 4.],
                                     [ 0.],
                                     [ 3.],
                                     [ 4.],
                                     [ 2.],
                                     [ 0.],
                                     [ 3.],
                                     [ 2.],
                                     [ 2.],
                                     [ 3.],
                                     [ 4.],
                                     [ 2.],
                                     [ 2.],
                                     [ 0.],
                                     [ 2.],
                                     [ 3.],
                                     [ 0.],
                                     [ 3.],
                                     [ 2.],
                                     [ 4.],
                                     [ 3.],
                                     [ 0.],
                                     [ 3.],
                                     [ 3.],
                                     [ 3.],
                                     [ 4.],
                                     [ 2.],
                                     [ 1.],
                                     [ 1.],
                                     [ 1.],
                                     [ 2.],
                                     [ 3.],
                                     [ 1.],
                                     [ 0.],
                                     [ 0.],
                                     [ 0.],
                                     [ 3.],
                                     [ 4.],
                                     [ 4.],
                                     [ 2.],
                                     [ 2.],
                                     [ 1.],
                                     [ 2.],
                                     [ 0.],
                                     [ 3.],
                                     [ 2.],
                                     [ 2.],
                                     [ 0.],
                                     [ 3.],
                                     [ 3.],
                                     [ 1.],
                                     [ 2.],
                                     [ 1.],
                                     [ 2.],
                                     [ 2.],
                                     [ 4.],
                                     [ 3.],
                                     [ 3.],
                                     [ 2.],
                                     [ 4.],
                                     [ 0.],
                                     [ 0.],
                                     [ 3.],
                                     [ 3.],
                                     [ 3.],
                                     [ 3.],
                                     [ 2.],
                                     [ 0.],
                                     [ 1.],
                                     [ 2.],
                                     [ 3.],
                                     [ 0.],
                                     [ 2.],
                                     [ 2.],
                                     [ 2.],
                                     [ 3.],
                                     [ 2.],
                                     [ 2.],
                                     [ 2.],
                                     [ 4.],
                                     [ 1.],
                                     [ 1.],
                                     [ 3.],
                                     [ 3.],
                                     [ 4.],
                                     [ 1.],
                                     [ 2.],
                                     [ 1.],
                                     [ 1.],
                                     [ 3.],
                                     [ 1.],
                                     [ 0.],
                                     [ 4.],
                                     [ 0.],
                                     [ 3.],
                                     [ 3.],
                                     [ 4.],
                                     [ 4.],
                                     [ 1.],
                                     [ 4.],
                                     [ 3.],
                                     [ 0.],
                                     [ 2.], ])

        self.assertTrue(np.array_equal(pred, expected_pred))


if __name__ == '__main__':
    unittest.main()
