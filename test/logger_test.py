import os
import sys
import unittest
import csv
from datetime import date
from ast import literal_eval
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution_guidance.logger import log_train, log_predict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'log')
LOG_PREFIX = 'test'

class LogTests(unittest.TestCase):
    def setUp(self):
        self.today = date.today()
        self.train_log_file = os.path.join(LOG_DIR, f"{LOG_PREFIX}-train-{self.today.year}-{self.today.month}.log")
        self.predict_log_file = os.path.join(LOG_DIR, f"{LOG_PREFIX}-predict-{self.today.year}-{self.today.month}.log")
        if os.path.exists(self.train_log_file):
            os.remove(self.train_log_file)
        if os.path.exists(self.predict_log_file):
            os.remove(self.predict_log_file)

    def test_train_log_creation(self):
        country = 'india'
        date_range = ('2017-11-29', '2019-05-24')
        metric = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        log_train(country, date_range, metric, runtime, model_version, model_version_note, test=True, prefix=LOG_PREFIX)
        self.assertTrue(os.path.exists(self.train_log_file))

    def test_train_log_content(self):
        country = 'india'
        date_range = ('2017-11-29', '2019-05-24')
        metric = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        log_train(country, date_range, metric, runtime, model_version, model_version_note, test=True, prefix=LOG_PREFIX)
        df = pd.read_csv(self.train_log_file)
        logged_metric = [literal_eval(i) for i in df['metric']][-1]
        self.assertEqual(metric, logged_metric)

    def test_predict_log_creation(self):
        y_pred = [0]
        y_proba = [0.6, 0.4]
        runtime = "00:00:02"
        model_version = 0.1
        country = "india"
        target_date = '2018-01-05'

        log_predict(country, y_pred, y_proba, target_date, runtime, model_version, test=True, prefix=LOG_PREFIX)
        self.assertTrue(os.path.exists(self.predict_log_file))

    def test_predict_log_content(self):
        y_pred = [0]
        y_proba = [0.6, 0.4]
        runtime = "00:00:02"
        model_version = 0.1
        country = "india"
        target_date = '2018-01-05'

        log_predict(country, y_pred, y_proba, target_date, runtime, model_version, test=True, prefix=LOG_PREFIX)
        df = pd.read_csv(self.predict_log_file)
        logged_y_pred = [literal_eval(i) for i in df['y_pred']][-1]
        self.assertEqual(y_pred, logged_y_pred)

if __name__ == '__main__':
    unittest.main(failfast=True)
