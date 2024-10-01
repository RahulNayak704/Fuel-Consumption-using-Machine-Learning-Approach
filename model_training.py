from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('data/auto-mpg.csv')

# Select features and target (after preprocessing)
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
target = 'mpg'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Train Support Vector Regression (SVR) model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

# Model Evaluation
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))
print("SVR MAE:", mean_absolute_error(y_test, svr_pred))

# R2 Score Evaluation
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("SVR R2:", r2_score(y_test, svr_pred))

# Save the best-performing model (Random Forest in this example)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
