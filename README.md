# IBM AI Workflow Capstone

## Overveiw
The IBM AI Workflow Capstone project involves building and deploying a time-series prediction model using machine learning algorithms. The model is trained on historical data to forecast trends for various countries.

## Project Structure

```
IBM_AI_Workflow_Capstone/
│
├── data/
│   └── cs-train/              # Training data directory
│   └── cs-production/         # Production data directory
│
├── solution_guidance/
│   ├── logger.py              # Logging utilities for model training and prediction
│   ├── cslib.py               # Custom helper functions for feature engineering and data preprocessing
│   └── model.py               # Core logic for training, predicting, and saving models
│   └── log/                   # Directory for logging outputs
│
├── test/                      # Unit tests for the solution
│   ├── model_test.py          # Test cases for the model functions
│   ├── api_test.py            # API test cases
│   └── logger_test.py         # Test cases for logging functionality
│
├── .gitignore                 # Ignore unnecessary files/folders for version control
├── README.md                  # Project documentation
└── Capstone_Project.ipynb     # Main script for running training and prediction
```

## Usage

To use this repository, download the data from the IBM AI Enterprise Workflow Capstone Project (https://github.com/aavail/ai-workflow-capstone) and store it under the folder 'data'.

### 1. Training the Model and Data Preprocessing
To run the model training and data preprocessing, open the provided  `Capstone_Project.ipynb` notebook and follow these steps:

**STEP 1: Data Ingestion and Preprocessing**

The notebook will load and preprocess the dataset. This includes converting date columns, extracting relevant features.

**Step 2: Exploratory Data Analysis (EDA)**

The notebook includes steps for performing exploratory analysis, such as revenue distribution by country and time period.

**Step 3: Feature Engineering**

In this step, the notebook processes the data into a time-series format and prepares it for training.

**Step 4: Model Training**
Train a RandomForestRegressor model using a pipeline with a scaler and RandomForest.

### 2. Prediction
You can use the trained model to make predictions by calling the model_predict() function.

### 3. Running Tests
To run the unit tests included in the test/ folder (e.g., model_test.py), use the following command:

```bash
python3 -m unittest discover -s test
```

This will run all the tests in the test directory to ensure the integrity of your model's functions.
