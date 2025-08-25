from xgboost import XGBRegressor

def train_xgboost(X_train, y_train, best_params):
    model = XGBRegressor(objective='reg:squarederror', **best_params)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_test):
    return model.predict(X_test)