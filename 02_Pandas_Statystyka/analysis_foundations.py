import pandas as pd
import numpy as np

# 1. WCZYTYWANIE I INFO
# Zakładamy, że plik jest w folderze data/
# Jeśli nie masz go lokalnie, odkomentuj linię z URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. CZYSZCZENIE (TWOJE DECYZJE)
# Usuwamy kolumny z ogromnymi brakami (>40%)
df = df.drop(columns=['Cabin'])

# Wypełniamy wiek MEDIANĄ, bo jest odporna na outlierów
# Uczyliśmy się, że średnia bywa przekłamana przez skrajne wartości.
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# 3. STATYSTYKA OPISOWA (Z-SCORE)
# Standaryzacja: Sprowadzamy wiek do skali, gdzie średnia = 0, a odchylenie = 1
# Dzięki temu model AI nie będzie 'myślał', że wiek jest ważniejszy niż cena biletu.
df['Age_ZScore'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

print("Podgląd wyczyszczonych danych z Z-Score: ")
print(df[['Age', 'Age_ZScore']].head())

# 4. AGREGACJA (GROUPBY)
# Sprawdźmy to, co widzieliśmy na boxplocie: średni wiek w klasach
print("\nŚredni wiek w zależności od klasy (Pclass):")
print(df.groupby('Pclass')['Age'].mean())