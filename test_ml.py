import os
import pandas as pd
import pytest
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split




class MyData: 
    def __init__(self, data, train, test, X_train, y_train, X_test, y_test, model, encoder, lb, preds, p, r, fb):
        self.data = data
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.encoder = encoder
        self.lb = lb
        self.train = train
        self.test = test
        self.X_test = X_test
        self.y_test = y_test     
        self.preds = preds  
        self.p = p
        self.r = r     
        self.fb = fb  

@pytest.fixture
def dataSet():
    project_path =  os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, train_size=0.7, test_size=0.3)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = 'salary'
    X_train, y_train, encoder, lb = process_data(
        X=train, 
        categorical_features=cat_features, 
        label=label
    )    
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    p, r, fb = compute_model_metrics(y_test, preds)

    return MyData(data, train, test, X_train, y_train, X_test, y_test, model, encoder, lb, preds, p, r, fb)


def test_if_model_is_of_correct_type(dataSet):
    """
    # Tests that the model is of typ XGBClassifier
    """
    assert type(dataSet.model) == XGBClassifier


def test_is_training_data_split_correctly(dataSet):
    """
    # Test that training and testing data sets match expectations
    """
    assert len(dataSet.train)+len(dataSet.test)==len(dataSet.data)


def test_x_y_train_correct_length(dataSet):
    """
    # Test to make sure X_train and y_train are of same length as training data
    """
    assert len(dataSet.X_train)==len(dataSet.y_train)==len(dataSet.train)

def test_predictions_correct_length(dataSet):
    """
    # Test to make sure X_train and y_train are of same length as training data
    """
    assert len(dataSet.preds)==len(dataSet.test)

def test_precision_recall_f1_greater_than_zero(dataSet):
    """
    # Test to ensure that metrics are returned appropriately
    """
    assert all([dataSet.p > 0, dataSet.r > 0, dataSet.fb > 0])