import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
speed = np.linspace(20, 160, 50)  
fuel_consumption = 5 + 0.01*(speed - 90)**2 + np.random.normal(0, 0.5, size=speed.shape)

X = speed.reshape(-1, 1)
y = fuel_consumption

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mse_list = []
mae_list = []
models = []
degrees = range(1, 8)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    mse_list.append(mse)
    mae_list.append(mae)
    models.append((model, poly))

plt.figure(figsize=(10, 4))
plt.plot(degrees, mse_list, 'o-', label='MSE')
plt.plot(degrees, mae_list, 's--', label='MAE')
plt.xlabel("Ступінь полінома")
plt.ylabel("Помилка")
plt.title("Залежність MSE/MAE від степеня полінома")
plt.legend()
plt.grid(True)
plt.show()

best_degree = degrees[np.argmin(mse_list)]
best_model, best_poly = models[np.argmin(mse_list)]

print(f" Найкращий ступінь полінома: {best_degree}")
print(f" MSE: {min(mse_list):.4f}")
print(f" MAE: {min(mae_list):.4f}")

X_all = np.linspace(20, 160, 300).reshape(-1, 1)
X_all_poly = best_poly.transform(X_all)
y_all_pred = best_model.predict(X_all_poly)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Виміри')
plt.plot(X_all, y_all_pred, color='red', label=f'Поліном степеня {best_degree}')
plt.title("Поліноміальна регресія: Витрати пального vs Швидкість")
plt.xlabel("Швидкість (км/год)")
plt.ylabel("Витрати пального (л/100 км)")
plt.legend()
plt.grid(True)
plt.show()

new_speeds = np.array([[35], [95], [140]])
new_speeds_poly = best_poly.transform(new_speeds)
predictions = best_model.predict(new_speeds_poly)

for v, pred in zip(new_speeds.flatten(), predictions):
    print(f"Прогнозовані витрати пального на {v} км/год: {pred:.2f} л/100 км")

