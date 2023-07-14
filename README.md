import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the car dataset
car_data = pd.read_csv('car_data.csv')

# Preprocess the data
label_encoder = LabelEncoder()
car_data['make'] = label_encoder.fit_transform(car_data['make'])

# Select the features (independent variables)
features = car_data[['make', 'mileage', 'age', 'horsepower']]

# Select the target variable (dependent variable)
target = car_data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Random Forest regression model with hyperparameter tuning
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Get feature importances
feature_importances = model.feature_importances_
print('Feature Importances:', feature_importances)

# Get the most important feature
most_important_feature = features.columns[feature_importances.argmax()]
print('Most Important Feature:', most_important_feature)

# Get descriptive statistics of the target variable
target_stats = target.describe()
print('Target Variable Statistics:')
print(target_stats)

# Get the average price of cars in the dataset
average_price = target.mean()
print('Average Price:', average_price)

# Make a single car price prediction
new_car = [[1, 50000, 3, 250]]  # Example values for make, mileage, age, and horsepower
predicted_price = model.predict(new_car)
print('Predicted Price for New Car:', predicted_price)

# Visualize the actual vs. predicted prices
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Car Prices')
plt.show()
