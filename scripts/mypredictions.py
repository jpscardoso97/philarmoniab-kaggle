import pandas as pd

def predict_subscriptions(model, features):
    print("Predicting subscriptions for test set with shape", features.shape)
    preds = model.predict(features)
    ids = features['ID']
    
    res = pd.DataFrame({'ID': ids, 'Predicted': preds})

    res.to_csv('../data/test_predictions.csv', index=False)
