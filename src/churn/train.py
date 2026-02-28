from sklearn.linear_model import LogisticRegression

def train_logreg_model(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """
    Train a logistic regression model
    """
    model = LogisticRegression(
        random_state=random_state, 
        max_iter=2000,
        class_weight='balanced',
        n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    return model