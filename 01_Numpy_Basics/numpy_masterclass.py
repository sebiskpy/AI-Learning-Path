import numpy as np

# 1. KREACJA I KSZTAŁT
# Zawsze sprawdzaj .shape, gdy coś nie działa!
arr = np.array([[1, 2, 3], [4, 5, 6]]) 

# 2. SLICING (Wycinanie) - Jak w Pythonie, ale wielowymiarowo
# Składnia: [wiersze, kolumny] -> [start:stop, start:stop]
first_row = arr[0, :]      # Pierwszy wiersz, wszystkie kolumny: [1, 2, 3]
last_column = arr[:, -1]   # Wszystkie wiersze, ostatnia kolumna: [3, 6]
sub_matrix = arr[0:2, 1:3] # Wiersze 0-1, kolumny 1-2: [[2, 3], [5, 6]]

# 3. BOOLEAN INDEXING (Filtrowanie "Maską")
# Genialne do czyszczenia danych bez pętli if/else
data = np.array([10, -5, 20, -1, 30])
mask = data > 0            # Tworzy tablicę: [True, False, True, False, True]
positive_data = data[mask] # Zwraca tylko [10, 20, 30]

# 4. BROADCASTING (Automatyczne dopasowanie)
# Dodajemy wektor (3,) do macierzy (2, 3). NumPy "rozmnoży" wektor dla każdego wiersza.
prices = np.array([[100, 200, 300], [400, 500, 600]])
tax_increase = np.array([10, 20, 30])
final_prices = prices + tax_increase 

# 5. RESHAPE (Zmiana struktury) - Kluczowe dla modeli ML
# Zamiana płaskiej listy 6 elementów na macierz 3x2
flat = np.array([1, 2, 3, 4, 5, 6])
matrix_3x2 = flat.reshape(3, 2) 
# Triki z -1: "NumPy, sam wylicz ile tu ma być wierszy, ale zrób 1 kolumnę"
column_vector = flat.reshape(-1, 1) # Shape (6, 1)

# 6. AGREGACJE I AXIS (Oś) - Gdzie liczymy?
# axis=0 -> pionowo (kolumny), axis=1 -> poziomo (wiersze)
matrix = np.array([[1, 2], [3, 4]])
col_sum = matrix.sum(axis=0) # [4, 6]
row_mean = matrix.mean(axis=1) # [1.5, 3.5]

# 7. DOT PRODUCT (Iloczyn skalarny) - Serce AI
# Mnożenie "Waga x Wejście"
inputs = np.array([0.5, 0.8])
weights = np.array([0.2, 0.7])
result = np.dot(inputs, weights) # Lub: inputs @ weights