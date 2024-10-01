import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('data/auto-mpg.csv')

# Handling missing values for 'horsepower' (numerical feature)
data['horsepower'] = data['horsepower'].replace('?', pd.NA).astype(float)
data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)

# Handling missing values for other numerical features (e.g., 'displacement', 'weight', etc.)
data['displacement'].fillna(data['displacement'].mean(), inplace=True)
data['weight'].fillna(data['weight'].mean(), inplace=True)
data['acceleration'].fillna(data['acceleration'].median(), inplace=True)

# Handling missing values for categorical features (e.g., 'cylinders', 'model year')
data['cylinders'].fillna(data['cylinders'].mode()[0], inplace=True)
data['model year'].fillna(data['model year'].mode()[0], inplace=True)

# Select features and target
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
target = 'mpg'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing completed, ready for modeling.")
