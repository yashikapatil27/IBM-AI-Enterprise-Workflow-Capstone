import os
import sys
import unittest
import requests
import json
import re

PORT = 8080
BASE_URL = f'http://0.0.0.0:{PORT}'

def server_is_up():
    try:
        response = requests.post(f'{BASE_URL}/ping')
        return response.status_code == 200
    except requests.ConnectionError:
        return False

class APITests(unittest.TestCase):
    def setUp(self):
        self.server_running = server_is_up()
        if not self.server_running:
            self.skipTest("Server not running")

    def test_model_training(self):
        payload = {'mode': 'test'}
        response = requests.post(f'{BASE_URL}/train', json=payload)
        result = re.sub(r'\W+', '', response.text)
        self.assertEqual(result, 'true')

    def test_empty_predict(self):
        response = requests.post(f'{BASE_URL}/predict')
        self.assertEqual(response.text.strip(), "[]")
        response = requests.post(f'{BASE_URL}/predict', json={"key": "value"})
        self.assertEqual(response.text.strip(), "[]")

    def test_single_country_prediction(self):
        data = {
            'query': {'country': 'united_kingdom', 'year': '2019', 'month': '06', 'day': '05'},
            'mode': 'test'
        }
        response = requests.post(f'{BASE_URL}/predict', json=data)
        predictions = json.loads(response.text)
        self.assertTrue(predictions['united_kingdom']['y_pred'])

    def test_multiple_country_prediction(self):
        data = {
            'query': {'country': 'united_kingdom,portugal', 'year': '2019', 'month': '06', 'day': '05'},
            'mode': 'test'
        }
        response = requests.post(f'{BASE_URL}/predict', json=data)
        predictions = json.loads(response.text)
        for country in ['united_kingdom', 'portugal']:
            self.assertTrue(predictions[country]['y_pred'])

    def test_log_file_retrieval(self):
        log_file = 'unittests-train-2020-6.log'
        response = requests.get(f'{BASE_URL}/logs/{log_file}')
        with open(log_file, 'wb') as file:
            file.write(response.content)
        self.assertTrue(os.path.exists(log_file))
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main(failfast=True)
