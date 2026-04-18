import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

nltk.download('stopwords')

# ======================
# LOAD DATA
# ======================
df = pd.read_csv('data_youtube.csv')
df['komentar'] = df['komentar'].astype(str)

# ======================
# HAPUS EMOJI
# ======================
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['clean'] = df['komentar'].apply(remove_emoji)

# ======================
# LOWERCASE
# ======================
df['clean'] = df['clean'].str.lower()

# ======================
# HAPUS SIMBOL
# ======================
df['clean'] = df['clean'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# ======================
# HAPUS SPASI BERLEBIH
# ======================
df['clean'] = df['clean'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# ======================
# STOPWORD
# ======================
stop_words = set(stopwords.words('indonesian'))

df['clean'] = df['clean'].apply(
    lambda x: ' '.join([w for w in x.split() if w not in stop_words])
)

# ======================
# STEMMING
# ======================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

df['clean'] = df['clean'].apply(lambda x: stemmer.stem(x))

# ======================
# LABELING
# ======================
def label_sentimen(text):
    if any(word in text for word in ['bagus','keren','seram','mantap','suka']):
        return 'positif'
    elif any(word in text for word in ['jelek','bosan','buruk','gagal']):
        return 'negatif'
    else:
        return 'netral'

df['sentimen'] = df['clean'].apply(label_sentimen)

# ======================
# SIMPAN
# ======================
df.to_csv('data_labeled.csv', index=False)

print("✅ Preprocessing selesai!")
print("Total data:", len(df))