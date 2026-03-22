import numpy as np


# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix = np.arange(1, 10).reshape(3, 3)

arr1 = np.arange(1, 10).reshape(3, 3)
arr2 = np.array([1,2,3])

# print(f"Original:\n{matrix}")
# print(f"Matrix 2:\n{matrix**2}")
print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")
print(f"Array 3:\n{arr1 +arr2}")

# To zadziała (Broadcasting)
a = np.ones((3, 3))
b = np.array([1, 2, 3])
print(a + b)
print("---")
# To WYWALI błąd (Niezgodne wymiary)
# c = np.array([1, 2])
# print(a + c) # Odkomentuj, żeby zobaczyć błąd w konsoli

arr3 = np.arange(1, 10).reshape(3, 3)
print(arr3)

arr4 = arr3[arr3 % 2 == 0 ]
arr5 = arr3[arr3 % 2 != 0 ]
print(f"Parzyste\n{arr4}\nShape: {arr4.shape}")
print(f"Nieparzyste\n{arr5}\nShape: {arr5.shape}\n")

print(f"Maska:\n{arr4}")

arr = np.arange(1, 10).reshape(3, 3)

# Krok 1: Maska (Tablica prawdy/fałszu)
mask = (arr % 2 == 0) 
print(f"Maska:\n{mask}") # To wciąż jest 3x3!

# Krok 2: Aplikacja maski
result = arr[mask] 
print(f"Wynik: {result}") # To już jest 1D

arr = np.arange(1, 10).reshape(3, 3)
print(f"{arr}\n---")

print(arr[1, :])
print(f"\n{arr[1:, 1:]}")