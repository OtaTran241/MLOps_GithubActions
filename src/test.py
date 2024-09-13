import unittest
import numpy as np
import pickle
import os

class TestModel(unittest.TestCase):
    def setUp(self):
        # Cập nhật đường dẫn đến model.pkl
        model_path = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([5.1, 3.5, 1.4, 0.2])
        prediction = self.model.predict(test_data.reshape(1, -1))
        self.assertIn(prediction[0], [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
