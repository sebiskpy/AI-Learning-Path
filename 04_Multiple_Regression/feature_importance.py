import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. TWOJE DANE
X = np.array([
    [10, 50,  2],
    [15, 100, 5],
    [20, 150, 7],
    [25, 300, 15],
    [30, 500, 20],
    [35, 600, 25]
])
y = np.array([120, 250, 400, 750, 1100, 1350])
feature_names = ['Godziny', 'Ads', 'Twitter']

# 2. STANDARYZACJA (Sprowadzamy wszystko do tej samej skali)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TRENOWANIE NA USTANDARYZOWANYCH DANYCH
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

# 4. PORÓWNANIE WAG
print("--- OBIEKTYWNA WAŻNOŚĆ CECH (Zstandaryzowane) ---")
for name, weight in zip(feature_names, model_scaled.coef_):
    print(f"Cecha: {name:8} | Waga: {weight:8.2f}")

print("\n(Te wagi mówią: o ile zmieni się 'y', gdy cecha zmieni się o 1 odchylenie standardowe)")