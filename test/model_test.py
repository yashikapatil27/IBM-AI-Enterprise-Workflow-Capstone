import unittest
import os
import sys
from datetime import date
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution_guidance.model import model_load, model_predict, model_train, nearest

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cs-train')

class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        self.model_path = os.path.join(MODEL_PATH, "ut-united_kingdom-0_1.joblib")
        self.country = 'united_kingdom'
        self.date_input = date.fromisoformat('2019-02-02')
        self.dates = ['2019-01-01', '2019-02-01', '2019-05-01', '2018-12-01']
        
    def test_model_training(self):
        _, models = model_load(prefix='ut', data_dir=DATA_PATH)  # Remove 'countries' argument
        model = list(models.values())[0]
        self.assertIn('predict', dir(model))
        self.assertIn('fit', dir(model))

    def test_model_loading(self):
        _, models = model_load(prefix='ut', data_dir=DATA_PATH, countries=[self.country])
        model = list(models.values())[0]
        self.assertIn('predict', dir(model))
        self.assertIn('fit', dir(model))

    def test_nearest_date_function(self):
        closest_date = nearest(self.dates, self.date_input)
        self.assertEqual(closest_date, '2019-02-01')

    def test_model_prediction(self):
        prediction_data = model_predict(self.country, '2018', '01', '05', prefix='ut')
        y_pred = prediction_data['y_pred']
        self.assertIsNotNone(y_pred[0])

if __name__ == '__main__':
    unittest.main(failfast=True)
