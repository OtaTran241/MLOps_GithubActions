import unittest
import pandas as pd
import pickle
import os

class TestModel(unittest.TestCase):
    def setUp(self):
        model_path = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.df_test = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'credit_score': [600, 700, 800],
            'country': ['France', 'Spain', 'Germany'],
            'gender': ['Male', 'Female', 'Male'],
            'age': [35, 45, 50],
            'tenure': [1, 2, 3],
            'balance': [1000.0, 1500.0, 2000.0],
            'products_number': [1, 2, 3],
            'credit_card': [1, 0, 1],
            'active_member': [1, 0, 1],
            'estimated_salary': [50000.0, 60000.0, 70000.0],
            'churn': [0, 1, 0]
        })

    def test_prediction(self):
        test_data = self.df_test.values.reshape(1, -1)

        prediction = self.model.predict(test_data)
        
        with open("single_test.txt", 'w') as outfile:
            outfile.write("Single test Results:\n")
            outfile.write(f"Input: {test_data.flatten().tolist()}\n")
            outfile.write(f"Prediction: {int(prediction[0])}\n")
        
        self.assertIn(prediction[0], [0, 1])

if __name__ == '__main__':
    unittest.main()
