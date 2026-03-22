import numpy as np

# --- 1. TWORZENIE I KSZTAŁT (SHAPE) ---
# Macierz to nie lista. To uporządkowana struktura w pamięci.
data = np.array([
    [10, 20], 
    [30, 40], 
    [50, 60]
])

print(f"Kształt macierzy: {data.shape}") # (3, 2) -> 3 wiersze, 2 kolumny

# --- 2. WEKTORYZACJA (BRAK PĘTLI FOR) ---
# W AI nigdy nie robimy: for x in data: x + 5. Robimy to tak:
plus_five = data + 5 
print(f"Dodawanie wektorowe:\n{plus_five}")

# --- 3. BROADCASTING ---
# NumPy 'rozmnaża' mniejszą tablicę, żeby pasowała do większej.
# Przykład: Odjęcie średniej od każdego wiersza (Normalizacja)
mean_values = np.mean(data, axis=0) # Średnia dla każdej kolumny
normalized = data - mean_values
print(f"Dane po odjęciu średniej (Broadcasting):\n{normalized}")

# --- 4. ILOCZYN SKALARNY (DOT PRODUCT) - SERCE NEURONU ---
# To jest operacja, która dzieje się wewnątrz sieci neuronowych:
# Wynik = (Wejście * Waga) + Bias
inputs = np.array([1, 2]) # Np. 2 cechy użytkownika
weights = np.array([0.5, -0.2]) # Wagi wyuczone przez model
bias = 0.1

prediction = np.dot(inputs, weights) + bias
print(f"Wynik prostego neuronu: {prediction}")