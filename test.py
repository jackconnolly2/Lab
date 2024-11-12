# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo

import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Step 1: Load and Explore the Dataset
# Fetch the dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Check for missing values
print("Missing values in features:")
print(X.isnull().sum())

# Step 2: Data Preprocessing (Feature Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape y to be a 1D array
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

# Step 4: Model 1 - Linear Regression (Baseline Model)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Model Performance:")
print(f"MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")

# Step 5: Model 2 - Multi-Layer Perceptron (MLP) Regression (Baseline)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42, learning_rate_init=0.01)
mlp_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_mlp = mlp_model.predict(X_test)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("\nBaseline MLP Model Performance:")
print(f"MAE: {mae_mlp:.2f}, MSE: {mse_mlp:.2f}, RMSE: {rmse_mlp:.2f}, R²: {r2_mlp:.2f}")

# Step 6: Hyperparameter Tuning for MLP
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh'],
    'max_iter': [1000, 2000]
}

grid_search = GridSearchCV(
    MLPRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# Step 7: Evaluate the Best Model
best_mlp = grid_search.best_estimator_
y_pred_best_mlp = best_mlp.predict(X_test)
mae_best_mlp = mean_absolute_error(y_test, y_pred_best_mlp)
mse_best_mlp = mean_squared_error(y_test, y_pred_best_mlp)
rmse_best_mlp = np.sqrt(mse_best_mlp)
r2_best_mlp = r2_score(y_test, y_pred_best_mlp)

print("\nOptimized MLP Model Performance:")
print(f"MAE: {mae_best_mlp:.2f}, MSE: {mse_best_mlp:.2f}, RMSE: {rmse_best_mlp:.2f}, R²: {r2_best_mlp:.2f}")

# Step 8: Build and Train a TensorFlow MLP Model (Deep Learning Approach)
model_tf = Sequential()
# Use Input layer to define the shape of the input
model_tf.add(Input(shape=(X_train.shape[1],)))  # X_train.shape[1] is the number of features
model_tf.add(Dense(100, activation='relu'))  # Hidden layer with 100 neurons
model_tf.add(Dense(50, activation='relu'))  # Hidden layer with 50 neurons
model_tf.add(Dense(1))  # Output layer for regression

# Compile the model
model_tf.compile(optimizer='adam', loss='mse')

# Train the model
history_tf = model_tf.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Evaluate TensorFlow Model
y_pred_tf = model_tf.predict(X_test).ravel()
mae_tf = mean_absolute_error(y_test, y_pred_tf)
mse_tf = mean_squared_error(y_test, y_pred_tf)
rmse_tf = np.sqrt(mse_tf)
r2_tf = r2_score(y_test, y_pred_tf)

print("\nTensorFlow MLP Model Performance:")
print(f"MAE: {mae_tf:.2f}, MSE: {mse_tf:.2f}, RMSE: {rmse_tf:.2f}, R²: {r2_tf:.2f}")

# Step 9: Visualize Results and Hyperparameter Impact

# Plot 1: MLP Model Loss Curve (Baseline MLP)
plt.figure(figsize=(10, 6))
plt.plot(mlp_model.loss_curve_)
plt.title("Baseline MLP Model Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot 2: Actual vs Predicted for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted for Linear Regression")
plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Actual vs Predicted for Baseline MLP
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_mlp, color='green', label='Baseline MLP Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted for Baseline MLP")
plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.legend()
plt.grid(True)
plt.show()

# Plot 4: Actual vs Predicted for Optimized MLP
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_mlp, color='purple', label='Optimized MLP Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted for Optimized MLP")
plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.legend()
plt.grid(True)
plt.show()

# Plot 5: Residuals for Linear Regression
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals_lr, color='blue', label='Linear Regression Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals for Linear Regression")
plt.xlabel("Predicted Compressive Strength")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.show()

# Plot 6: Residuals for Baseline MLP
residuals_mlp = y_test - y_pred_mlp
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_mlp, residuals_mlp, color='green', label='Baseline MLP Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals for Baseline MLP")
plt.xlabel("Predicted Compressive Strength")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.show()

# Plot 7: Residuals for Optimized MLP
residuals_best_mlp = y_test - y_pred_best_mlp
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best_mlp, residuals_best_mlp, color='purple', label='Optimized MLP Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals for Optimized MLP")
plt.xlabel("Predicted Compressive Strength")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.show()

# Plot 8: Hyperparameter Impact - Learning Rate vs Mean Test Score
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
for activation in results['param_activation'].unique():
    subset = results[results['param_activation'] == activation]
    plt.plot(subset['param_learning_rate_init'], -subset['mean_test_score'], label=f'Activation: {activation}')

plt.title("Hyperparameter Impact: Learning Rate vs Mean Test Score")
plt.xlabel("Learning Rate")
plt.ylabel("Mean Test Score (Negative MSE)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 9: Hyperparameter Impact - Hidden Layers vs Mean Test Score
plt.figure(figsize=(10, 6))
for size in results['param_hidden_layer_sizes'].unique():
    subset = results[results['param_hidden_layer_sizes'] == size]
    plt.plot(subset['param_max_iter'], -subset['mean_test_score'], label=f'Hidden Layers: {size}')

plt.title("Hyperparameter Impact: Hidden Layers vs Mean Test Score")
plt.xlabel("Max Iterations")
plt.ylabel("Mean Test Score (Negative MSE)")
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Compare Results
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'MLP (Baseline)', 'MLP (Optimized)', 'TensorFlow MLP'],
    'MAE': [mae_lr, mae_mlp, mae_best_mlp, mae_tf],
    'MSE': [mse_lr, mse_mlp, mse_best_mlp, mse_tf],
    'RMSE': [rmse_lr, rmse_mlp, rmse_best_mlp, rmse_tf],
    'R²': [r2_lr, r2_mlp, r2_best_mlp, r2_tf]
})

print("\nModel Comparison:")
print(comparison)
