import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data_labeled.csv')

# =========================
# FIX DATA (ANTI ERROR)
# =========================
# pastikan kolom clean ada
if 'clean' not in df.columns:
    print("❌ Kolom 'clean' tidak ditemukan!")
    exit()

# ubah semua jadi string & hilangkan NaN
df['clean'] = df['clean'].fillna('').astype(str)
df['sentimen'] = df['sentimen'].fillna('').astype(str)

# =========================
# WORDCLOUD SEMUA DATA
# =========================
all_text = ' '.join(df['clean'])

wc_all = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(all_text)

plt.figure()
plt.imshow(wc_all)
plt.axis('off')
plt.title("Wordcloud Semua Komentar")
plt.show()

# =========================
# FILTER PER SENTIMEN
# =========================
df_pos = df[df['sentimen'] == 'positif']['clean']
df_neg = df[df['sentimen'] == 'negatif']['clean']
df_net = df[df['sentimen'] == 'netral']['clean']

# =========================
# GABUNG TEKS
# =========================
text_pos = ' '.join(df_pos)
text_neg = ' '.join(df_neg)
text_net = ' '.join(df_net)

# =========================
# WORDCLOUD POSITIF
# =========================
wc_pos = WordCloud(width=800, height=400, background_color='white').generate(text_pos)

plt.figure()
plt.imshow(wc_pos)
plt.axis('off')
plt.title("Wordcloud Positif 😊")
plt.show()

# =========================
# WORDCLOUD NEGATIF
# =========================
wc_neg = WordCloud(width=800, height=400, background_color='white').generate(text_neg)

plt.figure()
plt.imshow(wc_neg)
plt.axis('off')
plt.title("Wordcloud Negatif 😡")
plt.show()

# =========================
# WORDCLOUD NETRAL
# =========================
wc_net = WordCloud(width=800, height=400, background_color='white').generate(text_net)

plt.figure()
plt.imshow(wc_net)
plt.axis('off')
plt.title("Wordcloud Netral 😐")
plt.show()

print("✅ Semua wordcloud berhasil tanpa error!")