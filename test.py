import numpy as np

# users = np.random.randint(100, 500, size=(7, 1))
# subs = np.random.randint(10, 50, size=(7, 1))
# revenue = np.random.uniform(100.0, 1000.0, size=(7, 1))

# data = np.column_stack((users, subs, revenue))
# summary = np.sum(data[:, 0], axis=0)
# avgConversionRate =  np.array(subs / users)
# maxRevenue = np.max(revenue)
# index = np.where(revenue == maxRevenue)

# print(f"Array data:\n{data}\n")
# print(f"Summary:\n{summary}\n")
# print(f"Average Conversion Rate:\n{avgConversionRate}\n")
# print(f"Max Revenue:\n{maxRevenue}\n")
# print(f"Index of Max Revenue:\n{index}\n")

# np.savetxt("sales.csv", data, delimiter=",", header="users,subs,revenue")

# 1. Generowanie danych (używamy nazw, które coś znaczą)
users = np.random.randint(100, 500, size=(7, 1))
subs = np.random.randint(10, 50, size=(7, 1))
revenue = np.random.uniform(100.0, 1000.0, size=(7, 1))

data = np.column_stack((users, subs, revenue))

# 2. Obliczenia (Pythonic & Clean)
total_revenue = data[:, 2].sum() # Sumujemy KOLUMNĘ 2 (revenue)
avg_conv = subs / users          # Nie potrzebujesz np.array(), to już jest array!
best_day_idx = np.argmax(data[:, 2]) # Czysty indeks najlepszego przychodu

meanRevenue = np.mean(revenue)
medianRevenue = np.median(revenue)

print(f"Data:\n{data}\n")
print(f"Total Revenue: ${total_revenue:.2f}")
print(f"Best Day Index: {best_day_idx}")
print(f"Mean Revenue: ${meanRevenue:.2f}")
print(f"Median Revenue: ${medianRevenue:.2f}")

# 3. Zapis do CSV (Ludzki format: %.2f oznacza 2 miejsca po przecinku)
np.savetxt("sales_clean.csv", data, delimiter=",", 
           header="users,subs,revenue", fmt="%.2f", comments='')