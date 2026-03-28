import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. PRZYGOTOWANIE (Dane muszą być macierzą 2D dla X!)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Godziny
y = np.array([10, 22, 31, 38, 52])           # Zarobek

print(f"X:\n{X}\n")
print(f"y:\n{y}\n")

# 2. TRENING (Dopasowanie linii)
model = LinearRegression()
model.fit(X, y)

# 3. INTERPRETACJA (Co wyliczył komputer?)
w = model.coef_[0]      # Waga
b = model.intercept_    # Bias
print(f"Model: y = {w:.2f}x + {b:.2f}")

# 4. PROGNOZA (Co będzie po 10 godzinach?)
future_x = np.array([[10]])
prediction = model.predict(future_x)
print(f"Prognoza dla 10h: {prediction[0]:.2f} zł")

# 5. WIZUALIZACJA
plt.scatter(X, y, color='blue', label='Historyczne dane')
plt.plot(X, model.predict(X), color='red', label='Linia trendu')
plt.legend()
plt.savefig("03_Linear_Regression/final_regression.png")