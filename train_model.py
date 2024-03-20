print("Imports...")
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

print("Get Data...")
# TODO: load the cencus.csv data
project_path =  os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)


print("Splitting Data...")
# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.3)

# DO NOT MODIFY
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

print("Process Training Data...")
# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
        X=data, 
        categorical_features=cat_features, 
        label=label
    )

print("Process Test Data...")
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

print("Train Model...")
# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)



print("Save Model and Encoder...")
# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)


print("Load Model..")
# load the model
model = load_model(
    model_path
) 

print("Infer Data...")
preds = inference(model, X_test)

print("Compute Metrics..")
# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")


print('Do slicers testing..')
# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test, 
            column_name=col, 
            slice_value=slicevalue, 
            categorical_features=cat_features, 
            label=label, 
            encoder=encoder, 
            lb=lb, 
            model=model      
        )
        test, col, slicevalue = compute_model_metrics(slicevalue, preds)
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

print("Complete!")