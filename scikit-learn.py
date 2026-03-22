import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Dane: Liczba reklam (X) i liczba sprzedanych subskrypcji (y)
# Reshape(-1, 1) jest konieczny, bo sklearn oczekuje macierzy 2D dla X
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 5, 7, 9, 11, 16, 15, 19, 21, 25])

# 2. Inicjalizacja i trenowanie modelu
model = LinearRegression()
model.fit(X, y) # Tu dzieje się magia optymalizacji

# 3. Przewidywanie dla nowej wartości (np. 12 reklam)
new_x = np.array([[12]])
prediction = model.predict(new_x)

print(f"Waga (w): {model.coef_[0]:.2f}")
print(f"Bias (b): {model.intercept_:.2f}")
print(f"Prognoza dla 12 reklam: {prediction[0]:.2f} subskrypcji")

# Wizualizacja
plt.scatter(X, y, color='blue', label='Dane historyczne')
plt.plot(X, model.predict(X), color='red', label='Model Regresji')
plt.legend()
plt.savefig("regression_test.png")