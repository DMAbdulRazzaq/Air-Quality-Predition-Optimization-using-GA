from sklearn.ensemble import RandomForestRegressor

def train_model(X, y, params):
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"]
    )
    model.fit(X, y)
    return model