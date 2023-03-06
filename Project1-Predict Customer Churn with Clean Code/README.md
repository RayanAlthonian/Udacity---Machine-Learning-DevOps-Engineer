# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project provides an end to end solution to a common problem in Machine Learning (ML) which is the prediction of credit card customer churn. It provides jupyter notebook that includes the typical data science pipeline consisting of EDA, model training and evaluation. However, the code is not production ready. Therefore, this project aims to make it production level by refactoring the code and following code best practices as well as implementing unit test to ensure that the functions work as intended.  

## Files and data description
Overview of the files and data present in the root directory. 

```
.
│   README.md                           # Provides project overview, and instructions to use the code
│   requirements.txt                    # Provides required packages to run the code
│   constants.py                        # Contains some variables such as cat_columns (used in churn_library.py & the testing script)
│   churn_notebook.ipynb                # Jupyter notebook that contains the whole data science pipeline (EDA, ML dev and ML Eval)
│   churn_library.py                    # Script that functions like the notebook, but has refactored code to ensure coding best practices
│   churn_script_logging_and_tests.py   # A set of unit tests to test the workings of key functions used in the script churn_library.py 
│
└───models                              # Stores pkl files of the trained models for later use
│       │   logistic_model.pkl              
│       │   rfc_model.pkl
│   
└───logs                                # Contains logs from the unit tests
│       │   churn_library.log                        
│
└───images
│       │  eda                          # Stores EDA results 
│       │  results                      # Stores model evaluation metrics and feature importance chart
│
└───data                                # Contains the data used in training the model

```

## Running Files

**Environment & Requirements Setup**

* Create an environment with python 3.8 
```
conda create -n MLOPsEnv python=3.8
```
* Activate the environment 
```
conda activate MLOPsEnv
```
* Install the required packages needed to run the code
```
pip install -r requirements.txt
```
**Script Execution**

* Run the script 
```
python churn_script.py
```
* Run the unit tests
```
python churn_script_logging_and_tests.py
```
