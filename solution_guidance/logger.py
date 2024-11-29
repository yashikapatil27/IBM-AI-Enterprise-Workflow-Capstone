import os
import time
import uuid
import csv
from datetime import date
import logging

# Constants
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Using Random Forest for time-series"
LOG_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'log')

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_log_filename(prefix, log_type):
    today = date.today()
    return os.path.join(LOG_DIR_PATH, f"{prefix}-{log_type}-{today.year}-{today.month}.log")

def log_to_csv(logfile, data, header, write_header=False):
    try:
        with open(logfile, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerow(data)
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")
        
def log_predict(country, y_pred, y_proba, target_date, runtime, model_version=MODEL_VERSION, test=False, prefix='model'):
    logfile = generate_log_filename(prefix, 'predict')

    header = ['unique_id', 'timestamp', 'country', 'y_pred', 'y_proba', 'target_date', 'model_version', 'runtime', 'mode']
    mode = 'test' if test else 'prod'
    data = [str(uuid.uuid4()), time.time(), country, y_pred, y_proba, target_date, model_version, runtime, mode]

    write_header = not os.path.exists(logfile)
    log_to_csv(logfile, data, header, write_header)

    logger.info(f"Prediction logged for {country} on {target_date}")

def log_train(country, date_range, metric, runtime, model_version=MODEL_VERSION, model_version_note=MODEL_VERSION_NOTE, test=False, prefix='model'):
    logfile = generate_log_filename(prefix, 'train')

    header = ['unique_id', 'timestamp', 'country', 'date_range', 'metric', 'model_version', 'model_version_note', 'runtime', 'mode']
    mode = 'test' if test else 'prod'
    data = [str(uuid.uuid4()), time.time(), country, date_range, metric, model_version, model_version_note, runtime, mode]

    write_header = not os.path.exists(logfile)
    log_to_csv(logfile, data, header, write_header)

    logger.info(f"Training log updated for {country} from {date_range[0]} to {date_range[1]}")

if __name__ == "__main__":
    log_predict('USA', 0.95, 0.78, '2024-12-01', '00:00:15')
    log_train('USA', ('2024-01-01', '2024-12-31'), {'rmse': 0.25}, '00:02:30')
