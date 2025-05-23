import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

def load_data():
    """Завантаження даних California housing"""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    return df

def preprocess_data(df):
    """Попередня обробка: масштабування та розділення"""
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model(input_shape):
    """Побудова моделі нейронної мережі"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Навчання моделі"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test):
    """Оцінка моделі"""
    loss, mae = model.evaluate(X_test, y_test)
    return loss, mae

def plot_loss(history):
    """Графік втрат"""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Крива навчання')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_test, y_pred):
    """Графік: справжні vs прогнозовані"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Справжня ціна')
    plt.ylabel('Прогнозована ціна')
    plt.title('Справжня vs Прогнозована ціна будинку')
    plt.grid(True)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = build_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, X_test, y_test)

    loss, mae = evaluate_model(model, X_test, y_test)
    print(f"📉 Test MSE: {loss:.4f}")
    print(f"📏 Test MAE: {mae:.4f}")

    plot_loss(history)

    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

