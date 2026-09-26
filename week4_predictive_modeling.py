import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Feature prep
X = pd.get_dummies(df[['distance_km', 'weight_kg', 'hub', 'day_of_week']],
                    columns=['hub'], drop_first=True)
y = df['delivery_time_hr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Ensemble model
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

print("Linear Regression:", metrics(y_test, lr_pred))
print("Random Forest:", metrics(y_test, rf_pred))

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("RF 5-fold CV RMSE:", -cv_scores.mean(), "+/-", cv_scores.std())

# Hyperparameter tuning
param_grid = {'n_estimators': [100, 200], 'max_depth': [4, 6, 8]}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                     scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# Feature importance
best_rf = grid.best_estimator_
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)
