import numpy as np
from sklearn.linear_model import LinearRegression

# DANE: [Godziny, Budżet Ads, Liczba postów na twiterze]
X = np.array([
    [10, 50],
    [15, 100],
    [20, 150],
    [25, 300],
    [30, 500]
])
print("Dane wejściowe (X):")
print(X)
# CEL: Liczba nowych użytkowników
y = np.array([120, 250, 400, 750, 1100])

# MODEL
model = LinearRegression()
model.fit(X, y)

# INTERPRETACJA WAG
# Dostaniemy dwie wagi: jedną dla godzin, drugą dla budżetu
w_godziny = model.coef_[0]
w_ads = model.coef_[1]
bias = model.intercept_

print(f"Waga dla godzin (w1): {w_godziny:.2f}")
print(f"Waga dla Ads (w2): {w_ads:.2f}")
print(f"Wartość startowa (b): {bias:.2f}")

# PROGNOZA: 
# Co jeśli w następnym tygodniu popracujesz 40h i wydasz 200$ na reklamy?
future_data = np.array([[40, 200]])
prediction = model.predict(future_data)

print(f"\nDane do prognozy (X_future): {future_data}")

print(f"\nPrognozowana liczba użytkowników: {prediction[0]:.0f}")