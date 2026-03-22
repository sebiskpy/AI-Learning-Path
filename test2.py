import numpy as np
import pandas as pd
# time = np.random.randint(1, 100, size=(10))

# mu = np.mean(time)
# sigma = np.std(time)

# # Obliczamy Z-Score dla każdego wyniku
# z_scores = (time - mu) / sigma

# muZscores = np.mean(z_scores)

# print(f"Średnia: {mu:.2f}s")
# print(f"Odchylenie (sigma): {sigma:.2f}s")
# print(f"Z-Scores: \n{z_scores}")
# print(f"Średnia Z-Score: {muZscores:.2f}")
# print(f"Time array:\n{time}\n")

# numbers = np.random.randn(1000)
# mean = np.mean(numbers)
# std_dev = np.std(numbers)   



# count = np.sum((numbers >= -1) & (numbers <= 1))
# percent = (count / len(numbers)) * 100

# print(percent)

# inputs = np.array([5, 2, 6])
# weights = np.array([0.2, 0.4, 0.6])

# output = np.dot(inputs, weights)  # To jest równoważne np.dot(inputs, weights)
# res2 = inputs @ weights
# print(f"Output: {output:.2f}")
# print(f"Output (using @): {res2:.2f}")


# 1. Tworzymy dane (podobnie jak obiekt w JS)
data = {
    'feature_size': [100, 200, 300],
    'price': [1500, 2500, 3500]
}

# 2. Tworzymy DataFrame - serce większości projektów ML
df = pd.DataFrame(data)

# 3. ML-owy trik: szybki podgląd statystyk (średnia, odchylenie itp.)
print(df.describe())