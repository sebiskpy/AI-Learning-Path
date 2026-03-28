import pandas as pd
import numpy as np

# 1. ŁADOWANIE I EKSPLORACJA
df = pd.read_csv("data/titanic.csv") # Zakładamy folder data/

# Szybki rzut oka na to, co mamy:
print(df.info())      # Typy danych i braki
print(df.describe())  # Średnie, min, max, kwartyle

# 2. CZYSZCZENIE DANYCH (Data Cleaning)
# Usuwanie kolumn z ogromnymi brakami
df.drop(columns=['Cabin'], inplace=True) 

# Inteligentne wypełnianie braków (naszą medianą!)
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# 3. FILTROWANIE (Odpowiednik Boolean Indexing z NumPy)
# Wybierz pasażerów z 1 klasy, którzy zapłacili > 100
rich_first_class = df[(df['Pclass'] == 1) & (df['Fare'] > 100)]

# 4. AGREGACJA (Groupby) - Klucz do wyciągania wniosków
# Czy klasa miała wpływ na przeżycie? (Średnia z kolumny 0/1 to % przeżycia)
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(f"Szansa na przeżycie wg klas:\n{survival_by_class}")

# 5. TWORZENIE NOWYCH CECH (Feature Engineering)
# Standaryzacja Ceny (Fare) za pomocą Z-Score
fare_mean = df['Fare'].mean()
fare_std = df['Fare'].std()
df['Fare_Z'] = (df['Fare'] - fare_mean) / fare_std

# 6. WYKRYWANIE OUTLIERÓW (Metoda IQR)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
limit_gorny = Q3 + 1.5 * IQR
outlierzy = df[df['Fare'] > limit_gorny]

# print(f"Outlierzy w Fare:\n{outlierzy[['Fare', 'Pclass', 'Survived']]}")        


# STANDARYZACJA DWÓCH CECH 

# Przykładowe dane
# data = {
#     'Fare': [500, 10, 50, 100, 20],
#     'Pclass': [1, 3, 2, 1, 3]
# }
# df = pd.DataFrame(data)

# # Standaryzacja Fare
# df['Fare_Z'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

# # Standaryzacja Pclass
# df['Pclass_Z'] = (df['Pclass'] - df['Pclass'].mean()) / df['Pclass'].std()

# print(df[['Fare_Z', 'Pclass_Z']])

# # Profesjonalnie (Scikit-Learn)
# #W prawdziwych projektach używamy StandardScaler, bo jest szybszy i łatwiejszy do wdrożenia w SaaS-ie.

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# # Wybieramy cechy do standaryzacji
# features = df[['Fare', 'Pclass']]

# # Trenujemy scaler i transformujemy dane
# scaled_features = scaler.fit_transform(features)

# # Wraca nam macierz NumPy, którą możemy wrzucić do DataFrame
# df_scaled = pd.DataFrame(scaled_features, columns=['Fare_scaled', 'Pclass_scaled'])

# print(df_scaled)
