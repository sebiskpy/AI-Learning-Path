import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. PRZYGOTOWANIE DANYCH (X musi być macierzą 2D)
# X = Godziny pracy, y = Liczba użytkowników
X = np.array([1, 2, 4, 8, 10, 15, 20, 25]).reshape(-1, 1)

# Scenariusz A: Normalny wzrost
y_normal = np.array([160, 210, 310, 505, 590, 850, 1120, 1350])

# Scenariusz B: Wzrost z "viralem" (Outlierem na końcu)
y_viral = np.array([160, 210, 310, 505, 590, 850, 1120, 5000])
ls

# 2. TRENOWANIE MODELI
model_a = LinearRegression().fit(X, y_normal)
model_b = LinearRegression().fit(X, y_viral)

# 3. PORÓWNANIE WYNIKÓW (WAGA I BIAS)
print("SCENARIUSZ A (Normalny):")
print(f"Waga (w): {model_a.coef_[0]:.2f} | Bias (b): {model_a.intercept_:.2f}")

print("\nSCENARIUSZ B (Viral/Outlier):")
print(f"Waga (w): {model_b.coef_[0]:.2f} | Bias (b): {model_b.intercept_:.2f}")

# 4. WIZUALIZACJA (Dlaczego outlier psuje model?)
plt.figure(figsize=(10, 5))
plt.scatter(X, y_viral, color='red', label='Dane z viralem')
plt.plot(X, model_b.predict(X), color='blue', label='Linia regresji (zepsuta)')
plt.title("Wpływ outliera na regresję liniową")
plt.savefig("03_Linear_Regression/regression_outlier.png")

# WNIOSEK DLA CIEBIE: 
# Zauważ, jak waga (w) w scenariuszu B drastycznie skoczyła. 
# Model próbuje "dogonić" ten jeden punkt 5000, przez co przestaje 
# dobrze opisywać pozostałe, normalne punkty.