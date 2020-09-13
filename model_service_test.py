import unittest
from emo_utils import read_glove_vecs
from model_service import ModelService
import numpy as np


class ModelServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.words, cls.index_to_words, cls.word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
        cls.model_service = ModelService()

    def test_sentence_to_avg(self):

        avg = self.model_service.sentence_to_avg("Morrocan couscous is my favorite dish", self.word_to_vec_map)
        print("avg = \n", avg)


if __name__ == '__main__':
    unittest.main()
