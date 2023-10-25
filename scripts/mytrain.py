from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np

# Let's scale the inputs to help it converge more easily
numerical_features = ['amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'num_subscriptions']

def get_best_model(X_train, y_train):
    kf = KFold(n_splits=10)
    models = [LogisticRegression(penalty='l2',C=1.0), RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1), SVC(kernel='linear', C=1), SVC(kernel='rbf', C=1)]
    model_names = ['LogisticRegression','Random Forest', 'Linear SVC', 'RBF SVC']
    model_aurocs = []

    for index, model in enumerate(models):
        aurocs = []
        for train_indexes, val_indexes in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_indexes], X_train.iloc[val_indexes]
            y_train_fold, y_val_fold = y_train.iloc[train_indexes], y_train.iloc[val_indexes]

            model.fit(X_train_fold, y_train_fold)

            aurocs.append(roc_auc_score(y_val_fold, model.predict(X_val_fold)))

        mean_auroc = np.mean(aurocs)
        print(model_names[index], " model AUROC: ", mean_auroc)
        model_aurocs.append(mean_auroc)
    
        #plot_decision_boundaries(X_train_scaled, y_train, model)

    best_model_index = np.argmax(model_aurocs)
    print("Best performance model is ", model_names[best_model_index])

    best_model = models[best_model_index]

    return best_model

def train_logistic_regression(X_train, y_train):
    display("Training Logistic Regression...")

    model = LogisticRegression(penalty='l2',C=1.0)
    model.fit(X_train,y_train)

    return model

def train_randomforest(X_train, y_train):
    display("Training Random Forest...")

    model = RandomForestClassifier(criterion='gini',max_depth=None, min_samples_leaf=3,n_estimators=1000,
                                 max_features=0.5,max_samples=0.5,random_state=1)
    
    model.fit(X_train,y_train)
    
    return model

# def get_best_knn(X_train, y_train):
#     best_acc = 0
#     best_model = None
#     optimal_n_neighbors = None

#     kf = KFold(n_splits=5)

#     for n_neighbors in [1,2,5,10]:
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)
#         accs = []
#         for train_indexes, val_indexes in kf.split(X_train):
#             X_train_fold,X_val_fold = X_train.iloc[train_indexes],X_train.iloc[val_indexes]
#             y_train_fold,y_val_fold = y_train.iloc[train_indexes],y_train.iloc[val_indexes]
#             model.fit(X_train_fold,y_train_fold)
#             pred = model.predict(X_val_fold)
#             acc = accuracy_score(y_val_fold,pred)
#             accs.append(acc)

#         if np.mean(accs) > best_acc:
#             best_acc = np.mean(accs)
#             optimal_n_neighbors = n_neighbors
#             best_model = model

#     print("Found best KNN with optimal neighbors:",optimal_n_neighbors)

#     return best_model