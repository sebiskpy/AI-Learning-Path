import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# DANE: [Godziny, Budżet Ads, Posty na Twitterze]
X = np.array([
    [10, 50,  2],
    [15, 100, 5],
    [20, 150, 7],
    [25, 300, 15],
    [30, 500, 20],
    [35, 600, 25]
])

# CEL: Liczba nowych użytkowników
y = np.array([120, 250, 400, 750, 1100, 1350])

# Nazwy naszych cech dla czytelności
feature_names = ['Godziny', 'Ads', 'Twitter']

# MODEL
model = LinearRegression()
model.fit(X, y)

# WYŚWIETLANIE WAG (Dynamiczne - nie na sztywno!)
print("--- ANALIZA WPŁYWU CECH ---")
for name, weight in zip(feature_names, model.coef_):
    print(f"Zmienna: {name:8} | Waga (Współczynnik): {weight:8.2f}")

print(f"\nBias (Punkt startowy): {model.intercept_:.2f}")

# PROGNOZA: 40h pracy, 400$ Ads, 30 postów
nowe_dane = np.array([[40, 400, 30]])
predykcja = model.predict(nowe_dane)

# Obliczamy jak dobry jest nasz model
r2 = model.score(X, y)
print(f"\nDokładność modelu (R-squared): {r2:.4f}")

print(f"\nPrzewidywany wynik: {predykcja[0]:.0f} użytkowników")

mse = mean_squared_error(y, model.predict(X))
rmse = np.sqrt(mse) # Pierwiastek z MSE daje błąd w tych samych jednostkach co 'y'

print(f"Średni błąd modelu: {rmse:.2f} użytkowników")