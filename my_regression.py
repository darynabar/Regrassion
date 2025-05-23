import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(loc=20, scale=10, size=n_samples)      
humidity = np.random.uniform(30, 90, size=n_samples)                   
hour = np.random.randint(0, 24, size=n_samples)                      
is_weekend = np.random.choice([0, 1], size=n_samples)                  

consumption = (
    10 + temperature * 0.5 + humidity * 0.2 +
    np.where((hour >= 18) & (hour <= 22), 15, 0) + 
    np.where(is_weekend == 1, -5, 0) +
    np.random.normal(0, 5, size=n_samples)
)

data = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "hour": hour,
    "is_weekend": is_weekend,
    "consumption": consumption
})

X = data[["temperature", "humidity", "hour", "is_weekend"]]
y = data["consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Справжнє споживання (кВт·год)")
plt.ylabel("Прогнозоване споживання (кВт·год)")
plt.title("Справжнє vs Прогнозоване споживання")
plt.grid(True)
plt.show()

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Середня відносна похибка: {mape:.2f}%")

