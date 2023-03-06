"""
Module Name: churn_script_logging_and_tests.py
Description: This module consists of several test functions used to test
the functionality of the functions in the module named churn_library.py

Author: Rayan Althonian
Date: 3 March 2023
"""
import os
from pathlib import Path
import logging
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
from constants import cat_columns

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function (Are the results saved?)
    '''
    df = import_data("./data/bank_data.csv")

    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error("Testing test_eda: This is not a dataframe")
        raise err

    path = Path("./images/eda")
    files = [
        "churned_customers_dist.png",
        "correlation_matrix.png",
        "Customer_age_distribution.png",
        "Customer_marital_status_distribution.png",
        "total_transcation_count_distribution.png"]

    for file in files:
        file_path = os.path.join(path, file)
        try:
            assert os.path.exists(file_path)
        except AssertionError as err:
            logging.error("ERROR: EDA results are not found")
            raise err
    logging.info("SUCCESS: EDA results successfully saved!")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = import_data("./data/bank_data.csv")
    response = 'Churn'
    new_cat_columns = [cat + "_" + response for cat in cat_columns]

    try:
        new_df = encoder_helper(df, cat_columns, response)
        assert set(new_cat_columns).issubset(new_df.columns)
        logging.info("SUCCESS: The encoder worked as expected")
    except AssertionError as err:
        logging.error("ERROR: The encoder does not work as intended")
        return err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = import_data("./data/bank_data.csv")
    df = encoder_helper(df, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info("SUCCESS: train and test splits are correct")
    except AssertionError as err:
        logging.error(
            "ERROR: The feature engineering function does not work :(")
        return err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = import_data("./data/bank_data.csv")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    train_models(X_train, X_test, y_train, y_test)

    path = Path("./models")
    models = ['logistic_model.pkl', 'rfc_model.pkl']
    models_paths = [os.path.join(path, model)for model in models]

    for model_path in models_paths:
        try:
            assert os.path.exists(model_path)
        except AssertionError as err:
            logging.error("ERROR: The models are not found")
            return err
    logging.info("SUCCESS: The models exist in the specified directory")


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
