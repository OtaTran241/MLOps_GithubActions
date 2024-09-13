import unittest
import numpy as np
import pickle

class TestModel(unittest.TestCase):
    def setUp(self):
        with open('../models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([5.1, 3.5, 1.4, 0.2])
        prediction = self.model.predict(test_data.reshape(1, -1))
        self.assertIn(prediction[0], [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
