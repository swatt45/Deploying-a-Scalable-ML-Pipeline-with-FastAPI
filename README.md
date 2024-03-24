# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## How to train / re-train
* Open a command prompt and type: python train_model.py
* Wait for the process to finish, and you should see the path of the outputs model and encoder files and the scores printed on the screen.
* model.pkl and encoder.pkl will be overwritten. These are curretly configured for the model directory.
* slice_output.txt lists the results of data slicing on the categorical values to see how individual slices score. It lists the slice and the scores.

# Data
* The data is in a CSV file in teh data folder: census.csv
* See the model_card.md in this directory. It discusses the structure and origin of the data.
* EDA.ipynb is a jupyter notebook used to perform EDA on the data. You can find this in the data file.
* In the data folder, a file called census_aequitas, was created in the EDA notebook. It us used to feed to Aequitas for bias analysis.
* Aequitas - The Bias Report.mhtml is the actual biase report generated during EDA. It is also embedded in the model_card.md.
* EDA uses ydata-profiling as well to output HTML that can be used to examine the data for abnormalities, patterns, and descriptive stats.

# Testing
* PyTest is used for running unit tests. 
* You can add unit tests to the test_ml.py.
* Run all unit tests using: pytest test_ml.py -v.

# API
* The API was built using FastAPI. 
* Host FastAPI where you want, and used the local_api.py client to make predictions using the model.