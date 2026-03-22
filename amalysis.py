import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 1. Wczytanie pliku


# 2. Filtrowanie (revenue > 500)
# best_days = df[df["revenue"] > 500]
# best_days = df.query("revenue > 500")

# 3. Sortowanie malejąco po subs
# best_days = best_days.sort_values(by="subs", ascending=False)

# 4. Zapis do pliku bez indeksu
# best_days.to_csv("vip_days.csv", index=False)


# print(new_df.to_string())


# df = pd.read_csv("sales.csv")
# df.iloc[0, 2] = np.nan # Psujemy dane

# # OPCJA A: Usuwanie (Najbezpieczniejsze, jeśli masz dużo danych)
# df_cleaned = df.dropna() 

# # OPCJA B: Imputacja średnią (Twoja propozycja)
# # UWAGA: Obliczamy średnią tylko dla kolumny revenue!
# mean_revenue = df['revenue'].mean()
# df['revenue'] = df['revenue'].fillna(mean_revenue)

# print(f"Wypełniono braki wartością: {mean_revenue:.2f}")
# print(df.head(3))

# data = {
#     'name' : ['Python', 'JavaScript', 'Java'],
#     'experience': [3, 6, 8],
#     'level' : [9, 7, 1]
# }

# df = pd.DataFrame(data)
# df = df.query("level > 5")
# print(df)

# print(df['experience'].mean())
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# print(df.isnull().sum())
# cabin_missing_percent = df["Cabin"].isnull().mean() * 100
# print(cabin_missing_percent)
# df.to_csv("titanic.csv", index=False) 
# Uzupełniamy wiek medianą (żeby nie było dziur w wykresie)
# df['Age'] = df['Age'].fillna(df['Age'].median())

# # Rysujemy histogram
# plt.figure(figsize=(10, 6))
# df['Age'].hist(bins=30, color='skyblue', edgecolor='black')
# plt.title("Rozkład wieku pasażerów Titanica")
# plt.xlabel("Wiek")
# plt.ylabel("Liczba osób")
# plt.savefig("titanic_age_hist.png")
# print("Wykres zapisany jako titanic_age_hist.png")

# Ustawiamy styl (Seaborn sprawia, że wykresy nie wyglądają jak z lat 90.)
# sns.set_theme(style="whitegrid")

# plt.figure(figsize=(10, 6))
# plt.ylim(0, 200)

# sns.boxplot(data=df, x='Survived', y='Fare', palette='Set2')

# plt.savefig("titanic_boxplot.png")
# print("Wykres zapisany!")

# 1. Tworzenie i kształt
data = np.array([[1, 2], [3, 4], [5, 6]]) # Shape (3, 2)
print(data)
# 2. Operacje wektorowe (zamiast pętli)
normalized = (data - np.mean(data)) / np.std(data)
print(normalized)
# 3. Iloczyn skalarny (Dot Product) - Twój pierwszy neuron
inputs = np.array([10, 20])
weights = np.array([0.8, 0.1])
bias = -5
output = (inputs @ weights) + bias # Wynik: (10*0.8 + 20*0.1) - 5 = 5.0
print(f"Output neuronu: {output}")