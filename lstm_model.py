import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('data_labeled.csv')

# hapus data kosong
df = df.dropna()

# =========================
# 2. TEXT & LABEL
# =========================
texts = df['clean']

# ubah label ke angka
labels = df['sentimen'].map({
    'negatif': 0,
    'netral': 1,
    'positif': 2
})

# =========================
# 3. TOKENIZING
# =========================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=100, padding='post')

y = np.array(labels)

# =========================
# 4. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. MODEL LSTM
# =========================
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================
# 6. TRAINING
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# 7. EVALUASI
# =========================
loss, acc = model.evaluate(X_test, y_test)

print("\n✅ Akurasi Model:", acc)

# =========================
# 8. SIMPAN MODEL
# =========================
model.save("model_lstm.h5")

print("✅ Model disimpan sebagai model_lstm.h5")


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# =========================
# PREDIKSI
# =========================
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['negatif','netral','positif']))