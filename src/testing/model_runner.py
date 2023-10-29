class ModelRunner:

    def __init__(self, features_builder):
        self.__features_builder = features_builder

    def test(self, model, test_data):
        X = self.__features_builder.build_features(test_data)
        X = X.drop(['account.id'], axis=1)

        preds = model.predict_proba(X)[:, 1]

        test_data['Predicted'] = preds

        test_data['ID'] = test_data['account.id']
        test_data.drop('account.id', axis=1, inplace=True)

        test_data[['ID', 'Predicted']].to_csv('../data/test_predictions.csv', index=False)

        print("Submission file created!")   
