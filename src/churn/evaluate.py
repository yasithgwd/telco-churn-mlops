from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate the binary classifier model and return mretrics w confusion matrics.
    """

    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    } 
    
    return metrics