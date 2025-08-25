from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, best_params):
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X_test):
    return model.predict(X_test)

# cố lên cố lênnn 1 2 1 2 
