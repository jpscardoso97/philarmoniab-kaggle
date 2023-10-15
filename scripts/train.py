from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

import pandas as pd
import numpy as np

# Let's scale the inputs to help it converge more easily
scaler = StandardScaler()
numerical_features = ['amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'num_subscriptions']

def get_best_model(X_train, y_train):
    kf = KFold(n_splits=10)
    best_knn = get_best_knn(X_train,y_train)
    models = [best_knn, SVC(kernel='linear', C=1), SVC(kernel='rbf', C=1), Perceptron()]
    model_names = ['KNN', 'Linear SVC', 'RBF SVC', 'Perceptron']
    model_aurocs = []

    for index, model in enumerate(models):
        aurocs = []
        for train_indexes, val_indexes in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_indexes], X_train.iloc[val_indexes]
            y_train_fold, y_val_fold = y_train.iloc[train_indexes], y_train.iloc[val_indexes]

            X_train_scaled_fold = scaler.fit_transform(X_train_fold[numerical_features])
            X_val_scaled_fold = scaler.transform(X_val_fold[numerical_features])
            model.fit(X_train_scaled_fold, y_train_fold)

            aurocs.append(roc_auc_score(y_val_fold, model.predict(X_val_scaled_fold)))

        mean_auroc = np.mean(aurocs)
        print(model_names[index], " model AUROC: ", mean_auroc)
        model_aurocs.append(mean_auroc)
    
        #plot_decision_boundaries(X_train_scaled, y_train, model)

    best_model_index = np.argmax(model_aurocs)
    print("Best performance model is ", model_names[best_model_index])

    best_model = models[best_model_index]

    X_train_scaled = scaler.fit_transform(X_train[numerical_features])
    best_model.fit(X_train_scaled, y_train)

    return best_model


def get_best_knn(X_train, y_train):
    best_acc = 0
    best_model = None
    optimal_n_neighbors = None

    kf = KFold(n_splits=5)

    for n_neighbors in [1,2,5,10]:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        accs = []
        for train_indexes, val_indexes in kf.split(X_train):
            X_train_fold,X_val_fold = X_train.iloc[train_indexes],X_train.iloc[val_indexes]
            y_train_fold,y_val_fold = y_train.iloc[train_indexes],y_train.iloc[val_indexes]
            X_train_scaled_fold = scaler.fit_transform(X_train_fold[numerical_features])
            X_val_scaled_fold = scaler.transform(X_val_fold[numerical_features])
            model.fit(X_train_scaled_fold,y_train_fold)
            accs.append(accuracy_score(y_val_fold,model.predict(X_val_scaled_fold)))

        if np.mean(accs) > best_acc:
            best_acc = np.mean(accs)
            optimal_n_neighbors = n_neighbors
            best_model = model

    print("Found best KNN with optimal neighbors:",optimal_n_neighbors)

    return best_model