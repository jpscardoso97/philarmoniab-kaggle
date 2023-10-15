import pandas as pd
import features as ft

def predict_subscriptions(model):
    test = pd.read_csv('../data/test.csv')

    features = ft.get_features(test)
    preds = model.predict(features)

    preds = preds['ID', 'Predicted']
    test.to_csv('../data/test_predictions.csv', index=False)
