from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import os, io, base64, json
from collections import Counter

app = Flask(__name__)

# =========================
# SETUP
# =========================
os.makedirs('static', exist_ok=True)

# Load data — ganti path sesuai file kamu
try:
    df = pd.read_csv('data_labeled.csv')
    df['sentimen'] = df['sentimen'].fillna('netral').astype(str).str.lower().str.strip()
    df['clean']    = df['clean'].fillna('').astype(str)
    DATA_LOADED = True
except Exception as e:
    print(f"[WARNING] data_labeled.csv tidak ditemukan: {e}")
    # Data dummy biar app tetap jalan
    df = pd.DataFrame({
        'sentimen': ['positif']*45 + ['negatif']*35 + ['netral']*20,
        'clean': ['bagus keren film seram'] * 45 +
                 ['jelek bosan buruk mengecewakan'] * 35 +
                 ['film biasa nonton'] * 20
    })
    DATA_LOADED = False


# =========================
# HELPER: fig → base64
# =========================
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor(), transparent=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# =========================
# GRAFIK: BAR DISTRIBUSI
# =========================
def buat_bar():
    BG = '#0a0a0f'
    counts = df['sentimen'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    color_map = {'positif': '#00e5a0', 'negatif': '#ff4444', 'netral': '#4488ff'}
    colors = [color_map.get(l, '#888') for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(BG)

    bars = ax.bar(labels, values, color=colors, width=0.5,
                  edgecolor='none', zorder=3)

    # Glow effect via shadow bars
    for bar, c in zip(bars, colors):
        ax.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
               width=bar.get_width() * 1.4, color=c, alpha=0.08,
               edgecolor='none', zorder=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}', ha='center', va='bottom',
                color='white', fontsize=11, fontweight='bold',
                fontfamily='monospace')

    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(colors='#666', labelsize=10)
    ax.set_xticklabels([l.upper() for l in labels],
                        color='#aaa', fontsize=9, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(axis='y', color='#ffffff10', linewidth=0.5, zorder=1)

    plt.tight_layout(pad=1.5)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# =========================
# GRAFIK: DONUT CHART
# =========================
def buat_donut():
    BG = '#0a0a0f'
    counts = df['sentimen'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    color_map = {'positif': '#00e5a0', 'negatif': '#ff4444', 'netral': '#4488ff'}
    colors = [color_map.get(l, '#888') for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
    ax.set_facecolor(BG)

    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.78,
        wedgeprops={'width': 0.55, 'edgecolor': BG, 'linewidth': 3}
    )
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(10); at.set_fontweight('bold')

    # Legend
    legend_elements = [mpatches.Patch(color=color_map.get(l, '#888'),
                                       label=f'{l.upper()}  {v:,}')
                       for l, v in zip(labels, values)]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, labelcolor='#aaa', fontsize=9)

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# =========================
# WORDCLOUD
# =========================
def buat_wordcloud_b64(sentimen=None):
    BG = '#0a0a0f'
    # color_func signature: (word, font_size, position, orientation, random_state, font_path)
    # WordCloud baru kirim semuanya sebagai keyword args — jangan pakai positional index
    import random as _rnd
    color_map_wc = {
        'positif': lambda word, font_size, position, orientation, random_state, **kw: '#00e5a0',
        'negatif': lambda word, font_size, position, orientation, random_state, **kw:
                       f'hsl({_rnd.randint(0, 20)}, 90%, {_rnd.randint(50, 70)}%)',
        'netral':  lambda word, font_size, position, orientation, random_state, **kw:
                       f'hsl(220, {_rnd.randint(50, 80)}%, {_rnd.randint(55, 70)}%)',
        None:      lambda word, font_size, position, orientation, random_state, **kw:
                       f'hsl({_rnd.randint(0, 360)}, 70%, 65%)',
    }

    if sentimen:
        subset = df[df['sentimen'] == sentimen]
    else:
        subset = df

    text = ' '.join(subset['clean'].dropna().astype(str))
    if not text.strip() or len(text.strip()) < 5:
        text = "tidak ada data tersedia"

    wc = WordCloud(
        width=900, height=420,
        background_color=BG,
        color_func=color_map_wc.get(sentimen, color_map_wc[None]),
        max_words=80,
        prefer_horizontal=0.8,
        max_font_size=90,
        random_state=42,
        collocations=False,
        margin=10,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# =========================
# STATISTIK RINGKAS
# =========================
def get_stats():
    total = len(df)
    counts = df['sentimen'].value_counts().to_dict()
    pos = counts.get('positif', 0)
    neg = counts.get('negatif', 0)
    net = counts.get('netral', 0)
    sentiment_score = round((pos - neg) / total * 100, 1) if total > 0 else 0
    return {
        'total': f'{total:,}',
        'positif': f'{pos:,}',
        'negatif': f'{neg:,}',
        'netral': f'{net:,}',
        'pos_pct': round(pos/total*100, 1) if total else 0,
        'neg_pct': round(neg/total*100, 1) if total else 0,
        'net_pct': round(net/total*100, 1) if total else 0,
        'sentiment_score': sentiment_score,
        'verdict': 'POSITIF' if sentiment_score > 10 else ('NEGATIF' if sentiment_score < -10 else 'MIXED'),
        'verdict_color': '#00e5a0' if sentiment_score > 10 else ('#ff4444' if sentiment_score < -10 else '#4488ff'),
    }


# =========================
# PREDIKSI (keyword-based / bisa diganti model)
# =========================
def predict_sentimen(text):
    t = str(text).lower()
    pos_kw = ['bagus','keren','seram','mantap','recommended','suka','keren',
               'terbaik','bangga','indah','menakjubkan','luar biasa','gokil',
               'kece','puas','senang','seru','asik','top','worth']
    neg_kw = ['jelek','bosan','buruk','kecewa','mengecewakan','payah','parah',
               'gak bagus','tidak bagus','sampah','buang','rugi','lebay',
               'membosankan','garing','zonk','waste','overrated']

    pos_score = sum(1 for k in pos_kw if k in t)
    neg_score = sum(1 for k in neg_kw if k in t)

    if pos_score > neg_score:
        label = 'positif'
        confidence = min(95, 60 + pos_score * 10)
    elif neg_score > pos_score:
        label = 'negatif'
        confidence = min(95, 60 + neg_score * 10)
    else:
        label = 'netral'
        confidence = 55

    return label, confidence


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    stats   = get_stats()
    bar_b64 = buat_bar()
    don_b64 = buat_donut()
    wc_all  = buat_wordcloud_b64(None)
    return render_template('index.html',
                           stats=stats,
                           bar_chart=bar_b64,
                           donut_chart=don_b64,
                           wordcloud_all=wc_all,
                           data_loaded=DATA_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Teks kosong'}), 400
    label, confidence = predict_sentimen(text)
    emoji_map = {'positif': '😍', 'negatif': '😤', 'netral': '😐'}
    color_map  = {'positif': '#00e5a0', 'negatif': '#ff4444', 'netral': '#4488ff'}
    return jsonify({
        'label':      label,
        'confidence': confidence,
        'emoji':      emoji_map[label],
        'color':      color_map[label],
        'bars': {
            'positif': confidence if label == 'positif' else max(5, 35 - confidence//3),
            'netral':  confidence if label == 'netral'  else 35,
            'negatif': confidence if label == 'negatif' else max(5, 35 - confidence//3),
        }
    })


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Tidak ada file'}), 400
    try:
        df_up = pd.read_csv(file)
    except Exception:
        return jsonify({'error': 'File CSV tidak valid'}), 400

    col = None
    for c in ['komentar', 'comment', 'text', 'teks', 'ulasan']:
        if c in df_up.columns:
            col = c; break
    if not col:
        col = df_up.columns[0]

    results, labels = [], []
    for text in df_up[col].fillna(''):
        label, conf = predict_sentimen(str(text))
        labels.append(label)
        results.append({
            'komentar':   str(text)[:100],
            'sentimen':   label,
            'confidence': f'{conf}%'
        })

    counts   = Counter(labels)
    total    = len(labels)
    summary  = {k: {'count': v, 'pct': round(v/total*100,1)} for k, v in counts.items()}

    return jsonify({'results': results[:50], 'summary': summary,
                    'total': total, 'column_used': col})


@app.route('/wordcloud/<sentimen>')
def wordcloud_route(sentimen):
    valid = ['positif', 'negatif', 'netral', 'all']
    if sentimen not in valid:
        return jsonify({'error': 'Invalid'}), 400
    b64 = buat_wordcloud_b64(None if sentimen == 'all' else sentimen)
    return jsonify({'image': b64})


if __name__ == '__main__':
    app.run(debug=True, port=5000)