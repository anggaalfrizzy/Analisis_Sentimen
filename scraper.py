from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

downloader = YoutubeCommentDownloader()

urls = [
    "https://youtu.be/8LIHcd7WfWI?si=lVEc6dJ2Qy-iKVQz",
    "https://youtu.be/Bfl1iMqd_ek?si=SagimasktUvMucWZ",
    "https://youtu.be/q0--C9_Gqsg?si=IFHmAE9tQMsRLeZE",
    "https://youtu.be/93c842501LM?si=V-Hs3P4cKRyX9MzT"
]

comments = []

for url in urls:
    print(f"Ambil komentar dari: {url}")
    count = 0
    
    for comment in downloader.get_comments_from_url(url):
        comments.append(comment['text'])
        count += 1
        
        if count >= 1000:
            break

print(f"Total komentar: {len(comments)}")

df = pd.DataFrame(comments, columns=['komentar'])
df['platform'] = 'youtube'

df.to_csv('data_youtube.csv', index=False)

print("✅ Berhasil simpan ke CSV!")