# Model Card

## Model Details
The data comes from the UC Irvine ML Repository and can be found here: https://archive.ics.uci.edu/dataset/20/census+income. The dataset is s multi-variate dataset of categorical and numeric features. It also states whether the individual had a salary greater than $50k annually.

## Intended Use

This SALARY classifier predicts salaries greater than 50k annually based several demographic features about a person, such as their age, race, native-country, and more. Once new samples are fed in, a prediction can be made from the model. The idea is to classify those individuals.

## Training Data
* Algorithm: XGBoost Classifier.
* Library: dmlc.
* Parameters: Default.
* Documentation URL: https://xgboost.readthedocs.io/en/stable/index.html.
* Seventy percent of the data was reserved for training.

## Evaluation Data
* Thirty percent of the data was reserved for testing. 
* Predictions were made from data not seen by testing. 
* Precision, Recall, and F1 were computed from the test predictions to evaluate the model.

## Metrics
Measures used to score this model's performance and results:

* Precision - What proportion of positive identifications was correct?
    * $True\:positives / (True\:positives + False\:positives)$
* Recall - What proportion of actual positives was identified correctly?
    * $True\:positives / (True\:positives + False\:negatives)$
* F1 - $F\beta$ scorer with beta equal to 1.
    * $F1 = ((1+\beta^2)*precision * recall) / \beta^2 * precision + recall$

The model scored the following:
* Precision: 0.7600
* Recall: 0.6651
* F1: 0.7094

This is a fairly balanced set of scores. The model looks pretty good. and the AUC is fairly large.

![image info](/data/ROC.png)

## Ethical Considerations
* According to Aequitas there are biases in race, sex, and native-country. 
* False Omission Rate Parity failed for these attributes. The reference group was white males native to the United States.
    * For race (with reference group as White)
        * Black with 0.48X Disparity
        * Other with 0.36X Disparity
        * Amer-Indian-Eskimo with 0.45X Disparity

    * For sex (with reference group as Male)
        * Female with 0.36X Disparity

    * For native-country (with reference group as United-States)
        * France with 1.68X Disparity
        * El-Salvador with 0.35X Disparity
        * Holand-Netherlands with 0.00X Disparity
        * Jamaica with 0.50X Disparity
        * Italy with 1.39X Disparity
        * Vietnam with 0.30X Disparity
        * Yugoslavia with 1.53X Disparity
        * Germany with 1.31X Disparity
        * Honduras with 0.31X Disparity
        * Haiti with 0.37X Disparity
        * Trinadad&Tobago with 0.43X Disparity
        * Canada with 1.31X Disparity
        * Thailand with 0.68X Disparity
        * Laos with 0.45X Disparity
        * England with 1.36X Disparity
        * Nicaragua with 0.24X Disparity
        * Dominican-Republic with 0.12X Disparity
        * Iran with 1.70X Disparity
        * Philippines with 1.25X Disparity
        * Taiwan with 1.60X Disparity
        * Ecuador with 0.58X Disparity
        * Japan with 1.57X Disparity
        * Puerto-Rico with 0.43X Disparity
        * Columbia with 0.14X Disparity
        * Outlying-US(Guam-USVI-etc) with 0.00X Disparity
        * Guatemala with 0.19X Disparity
        * Portugal with 0.44X Disparity
        * India with 1.63X Disparity
        * Mexico with 0.21X Disparity
        * Peru with 0.26X Disparity
        * Cambodia with 1.50X Disparity

* The full Aequitas report is located [here](./data/Aequitas%20-%20The%20Bias%20Report.mhtml) as an MHTML file. 
* This model uses the default parameters; however, it was not optimized for the best accuracy, only for best performance. 
## Caveats and Recommendations
For future versions of this data, we would like to obtain more diverse data that is more representative of other demographics. Further, it is recommended to find the optimal algorithm and training parameters. While a model was selected during EDA, a more thorough and automated approach to reach a decision on the correct algorithm and parameters could be warranted. This may be a good set of data and algorithm to use for targeting sales at people who can afford it, but basing any decisions affecting the lives of individuals, such as healthcare or government spending decisions, should be approached with caution. 

