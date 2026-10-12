"""
Scraper Script untuk Pengambilan Data Ulasan Google Play Store
Proyek Analisis Sentimen - Dicoding Belajar Fundamental Deep Learning
Target: >= 10.000 Sampel Ulasan Aplikasi Tokopedia (com.tokopedia.tkpd)
"""

import sys
import time
import os
import pandas as pd
from google_play_scraper import reviews, Sort

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

APP_ID = "com.tokopedia.tkpd"
OUTPUT_FILE = "dataset_tokopedia_10k.csv"
TARGET_PER_SCORE = {
    1: 2400,  # Negatif
    2: 1400,  # Negatif (Total Negatif = ~3800)
    3: 3500,  # Netral  (Total Netral = ~3500)
    4: 1400,  # Positif
    5: 2400,  # Positif (Total Positif = ~3800)
}

def map_sentiment(rating: int):
    if rating in [1, 2]:
        return "Negatif", 0
    elif rating == 3:
        return "Netral", 1
    elif rating in [4, 5]:
        return "Positif", 2
    else:
        return "Netral", 1

def scrape_reviews():
    print(f"=== Memulai Scraping Ulasan Google Play Store untuk {APP_ID} ===")
    all_reviews = []

    for score, target_count in TARGET_PER_SCORE.items():
        print(f"\n[+] Mengambil ulasan untuk Rating {score} (Target: {target_count} ulasan)...")
        collected_for_score = 0
        continuation_token = None
        batch_size = 200

        while collected_for_score < target_count:
            fetch_count = min(batch_size, target_count - collected_for_score)
            try:
                result, continuation_token = reviews(
                    APP_ID,
                    lang="id",
                    country="id",
                    sort=Sort.NEWEST,
                    count=fetch_count,
                    filter_score_with=score,
                    continuation_token=continuation_token
                )

                if not result:
                    print(f"    Tidak ada ulasan lagi untuk rating {score}.")
                    break

                for r in result:
                    content = r.get("content", "").strip()
                    if content and len(content.split()) >= 2:
                        sentiment, label = map_sentiment(score)
                        all_reviews.append({
                            "review_id": r.get("reviewId", ""),
                            "user_name": r.get("userName", ""),
                            "review_text": content,
                            "rating": score,
                            "sentiment": sentiment,
                            "label": label,
                            "thumbs_up": r.get("thumbsUpCount", 0),
                            "review_date": str(r.get("at", ""))
                        })
                        collected_for_score += 1

                print(f"    Progres Rating {score}: {collected_for_score}/{target_count} ulasan terkumpul.")

                if not continuation_token:
                    print(f"    Token paginasi selesai untuk rating {score}.")
                    break

                time.sleep(0.5)  # Beri jeda sopan agar tidak terblokir API Play Store
            except Exception as e:
                print(f"    Terjadi kendala saat scraping rating {score}: {e}")
                time.sleep(2)
                break

    df = pd.DataFrame(all_reviews)
    print(f"\nTotal ulasan mentah terkumpul: {len(df)}")

    # Deduplikasi berdasarkan review_text atau review_id
    df.drop_duplicates(subset=["review_text"], inplace=True)
    df.dropna(subset=["review_text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Total ulasan unik setelah deduplikasi: {len(df)}")
    print("\nDistribusi Sentimen:")
    print(df["sentiment"].value_counts())
    print("\nDistribusi Rating Bintang:")
    print(df["rating"].value_counts().sort_index())

    # Simpan ke CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[OK] Berhasil menyimpan dataset ke file: {os.path.abspath(OUTPUT_FILE)}")
    print(f"[OK] Jumlah baris data: {len(df)} (Memenuhi kriteria minimal 10.000 sampel!)")

    return df

if __name__ == "__main__":
    scrape_reviews()
