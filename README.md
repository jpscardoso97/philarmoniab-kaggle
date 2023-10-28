# philarmoniab-kaggle
Kaggle - Phillarmonia Baroque Orchestra &amp; Chorale 2014-15 Subscription Predictions

This repository supports all the steps of the development of an assignment to tackle the "Subscription Predictions for PBO" Kaggle challenge.

The project was approached following these steps:
- Brainstorm possible features
- Build basic models with 1 or 2 of the features considered most important using Logistic Regression, SVM and KNN
- Develop data pipeline to:
   - Load data 
   - Clean data
   - Generate features and train Random Forest Classifier with tuned hyperparameters
   - Run model on test data  

## Determine possible features
The first step was to determine which features could be used to predict the subscriptions. The features were divided in 3 categories.

### Demographic
The demographic features are the ones that describe the characteristics of the customer (location, age, etc.). From this category, only the zipcode was used as a feature.

### Behavioral
This type of feature is associated with how the account holder has behaved in the past (previous purchases, etc.). From the initial point of the project, this was the most valued category of features. The features that were used from this category were:
- Similarity between the conductors of previous subscription's season and 2014-15 season 
- Total amount spent on subscriptions (relates number of subscriptions and the price level)
- Ticket/subscription price level
- Subscription section (e.g.: Premium Orchestra, etc.)
- Season in which the customer bought the subscription
- Package of the subscription (e.g.: Quartet, Full, etc.)
- Subscription tier
- Number of seats per subscription

### Seasonal
The seasonal features are the ones that describe the season in which the customer is buying the subscription. From this category, the following features were used:
- Average of the seasons in which the customer bought the subscription
- Number of seats bought this season (2013)

## Data pipeline 
The pipeline is composed by 3 main steps:
- Load data
- Clean data
- Generate features
- Fine tune and train model
- Test model
- Generate submission file

### Clean data
The data was cleaned by either removing the rows with missing values or filling these values with 0 or other values like the mean of all data points.  
Categorial data was transformed into numerical data using one-hot encoding and specific mappings.

### Generate features
The final set of features is a combination of the demographic, behavioral and seasonal features described above. The choice took into consideration the correlation between features which led to the removal of a few highly correlated ones. The usage of a Random Forest Classifier also helped to determine the most important features and to remove the ones that were not contributing to the model. The ranking of the features was consistently analysed during the development of the project to verify which features were contributing to the model.

## Perform modeling
During the modeling phase, the following algorithms were used:
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest Classifier

The first three were tested during the initial phase of the project when there was not a big number of features. To optimize metrics cross-validation was used to adjust hyperparameters. When the number of features increased, the Random Forest algorithm was used because it is more robust to overfitting and allows better interpretability.

### Random forest classifier hyperparameters tuning
To optimize the hyperparameters of the Random Forest algorithm, a grid search was used. The following hyperparameters were tuned:
- n_estimators
- max_features
- max_depth
- min_samples_split
- min_samples_leaf

### Calculate predictions on test data
During the analysis of different models it was used accuracy during cross validation and AUROC for validation and to measure the performance of the model.
The data pipeline saves the trained model and uses it by default (instead of running training every time the pipeline is run). The model is saved in the `saved_models` folder.

## How to run the code
To run the pipeline simply run main.py