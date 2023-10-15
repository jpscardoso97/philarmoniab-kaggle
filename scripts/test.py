from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def get_test_score(X_test, y_test, best_model):
    scaler = StandardScaler()
    X_test_scaled = scaler.transform(X_test)
    preds = best_model.predict(X_test_scaled)
    auroc_score = roc_auc_score(y_test,preds)

    print("Auroc Score:",auroc_score)
