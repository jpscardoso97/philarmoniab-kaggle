# add necessary imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

class RandomForestTrainer:

    def __init__(self, features_builder):
        self.__features_builder = features_builder

    def train(self, train_data):
        print("Building training features...")

        X = self.__features_builder.build_features(train_data)
        y = X['label']
        X = X.drop(['account.id', 'label'], axis=1)

        print(X.columns)

        X_train,X_val,y_train,y_val = train_test_split(X, y, random_state=0, test_size=0.2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        params = {'min_samples_leaf':[1,3,10],'n_estimators':[100,1000],
            'max_features':[0.1,0.5,1.],'max_samples':[0.5,None]}

        m = RandomForestClassifier()
        grid_search = GridSearchCV(m,params,cv=3)
        grid_search.fit(X_train_scaled,y_train)

        bp = grid_search.best_params_

        print("Training Random Forest Classifier model with best parameters: ", bp)

        rf_model = RandomForestClassifier(criterion='gini',max_depth=None, min_samples_leaf=bp['min_samples_leaf'],n_estimators=bp['n_estimators'],
                                    max_features=bp['max_features'],max_samples=bp['max_samples'])
    
        rf_model.fit(X_train_scaled, y_train)
        
        val_preds = rf_model.predict_proba(X_val_scaled)[:, 1]

        print('Validation set AUROC is {:.3f}'.format(roc_auc_score(y_val,val_preds)))

        # Train model on entire dataset
        rf_model.fit(X, y)
        preds = rf_model.predict_proba(X)[:, 1]
        print('AUROC on entire training set: {:.3f}'.format(roc_auc_score(y,preds)))

        return rf_model
