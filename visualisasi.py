import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data_labeled.csv')

# =========================
# BERSIHKAN DATA
# =========================
df['sentimen'] = df['sentimen'].fillna('netral').astype(str)

# =========================
# HITUNG JUMLAH
# =========================
jumlah = df['sentimen'].value_counts()
print("Jumlah Sentimen:")
print(jumlah)

# =========================
# HITUNG PERSENTASE
# =========================
persen = df['sentimen'].value_counts(normalize=True) * 100
persen = persen.round(2)

print("\nPersentase Sentimen:")
for i in persen.index:
    print(f"{i}: {persen[i]}%")

# =========================
# GRAFIK BAR (JUMLAH)
# =========================
plt.figure()

ax = sns.countplot(x='sentimen', data=df)

# label angka
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

plt.title("Distribusi Sentimen (Jumlah)")
plt.xlabel("Kategori Sentimen")
plt.ylabel("Jumlah Komentar")

plt.show()

# =========================
# PIE CHART (PERSENTASE)
# =========================
plt.figure()

plt.pie(
    persen,
    labels=persen.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Persentase Sentimen Netizen")

plt.show()

# =========================
# BAR CHART PERSENTASE
# =========================
plt.figure()

persen.plot(kind='bar')

# label %
for i, v in enumerate(persen):
    plt.text(i, v, f"{v}%", ha='center')

plt.title("Persentase Sentimen (%)")
plt.xlabel("Kategori Sentimen")
plt.ylabel("Persentase")

plt.xticks(rotation=0)

plt.show()

print("\n✅ Visualisasi lengkap berhasil!")