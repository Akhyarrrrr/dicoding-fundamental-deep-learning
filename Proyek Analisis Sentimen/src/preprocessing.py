"""
Modul Preprocessing Teks Bahasa Indonesia
Proyek Analisis Sentimen - Dicoding Belajar Fundamental Deep Learning
"""

import re
import string

# Kamus Normalisasi Kata Slang / Informal ke Kata Baku Bahasa Indonesia
SLANG_DICTIONARY = {
    "bgt": "banget", "bngt": "banget", "buanget": "banget",
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "g": "tidak", "kagak": "tidak",
    "yg": "yang", "yng": "yang", "y": "yang",
    "tp": "tapi", "tpi": "tapi",
    "dgn": "dengan", "dg": "dengan",
    "klo": "kalau", "kalo": "kalau", "klw": "kalau",
    "bgs": "bagus", "baguuus": "bagus", "bgus": "bagus",
    "mantap": "bagus", "mantul": "bagus", "top": "bagus", "jos": "bagus", "keren": "bagus", "oke": "bagus", "ok": "bagus",
    "blm": "belum", "blom": "belum", "belom": "belum",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "da": "sudah",
    "makasih": "terima kasih", "thx": "terima kasih", "thanks": "terima kasih", "tks": "terima kasih", "mksh": "terima kasih",
    "apk": "aplikasi", "app": "aplikasi", "apli": "aplikasi",
    "lemot": "lambat", "lola": "lambat", "lelet": "lambat",
    "eror": "rusak", "error": "rusak", "crash": "rusak", "bug": "rusak",
    "nyesel": "menyesal", "kecewa": "kecewa",
    "batal": "batal", "cancel": "batal",
    "ongkir": "ongkos kirim", "promo": "promosi", "diskon": "diskon",
    "seller": "penjual", "buyer": "pembeli", "cs": "layanan pelanggan",
    "toped": "tokopedia", "tokped": "tokopedia",
    "brg": "barang", "barng": "barang",
    "sy": "saya", "ak": "saya", "aku": "saya", "gue": "saya", "gw": "saya",
    "km": "kamu", "lu": "kamu", "lo": "kamu",
    "bisaa": "bisa", "bs": "bisa",
    "knp": "kenapa", "knpa": "kenapa", "np": "kenapa",
    "gmn": "bagaimana", "gmana": "bagaimana",
    "bener": "benar", "beneran": "benar",
    "cepet": "cepat", "cpt": "cepat",
    "murah": "murah", "mumer": "murah", "mahal": "mahal",
    "ribet": "rumit", "susah": "sulit", "gampang": "mudah",
    "update": "pembaruan", "perbarui": "pembaruan",
    "bintang": "bintang", "bintng": "bintang",
    "packing": "pengemasan", "kirim": "pengiriman", "kiriman": "pengiriman",
}

# Stopwords Selektif (Menghapus kata hubung netral, mempertahankan kata negasi & sentimen)
STOPWORDS_SELECTIVE = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "pada", "adalah",
    "untuk", "dengan", "oleh", "sebagai", "seperti", "karena", "sehingga", "maka",
    "agar", "supaya", "jika", "kalau", "bila", "bahwa", "pun", "saja", "sih",
    "nih", "tuh", "dong", "deh", "kok", "lah", "kah", "ya", "yah", "kan",
    "para", "suatu", "ia", "dia", "mereka", "kita", "kami", "anda",
    "tentang", "antara", "serta", "kemudian", "lalu", "namun"
}

def clean_text(text: str) -> str:
    """
    Membersihkan dan menormalisasi teks ulasan:
    1. Lowercase (case folding)
    2. Menghapus URL, Mention (@user), Hashtag (#tag)
    3. Menghapus Emoji dan simbol non-ASCII
    4. Menghapus tanda baca dan angka
    5. Menghapus karakter berulang (> 2 karakter berurutan)
    6. Normalisasi slang words
    7. Menghapus stopwords selektif
    8. Normalisasi spasi
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Hapus Mention & Hashtag
    text = re.sub(r"[@#]\w+", " ", text)

    # 4. Hapus Emoji & karakter non-ASCII
    text = text.encode("ascii", "ignore").decode("utf-8")

    # 5. Hapus angka & tanda baca
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. Reduksi karakter berulang
    # Konsonan berulang (misal: "mantulll" -> "mantul", "bgttt" -> "bgt")
    text = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", text)
    # Vokal berulang >= 3 (misal: "baguuus" -> "bagus", tapi "saat"/"maaf" tetap terjaga)
    text = re.sub(r"([aeiou])\1{2,}", r"\1", text)

    # 7. Tokenisasi kata untuk normalisasi slang & filter stopword selektif
    words = text.split()
    normalized_words = []
    for word in words:
        # Ganti dengan kata baku jika ada di slang dictionary
        norm_word = SLANG_DICTIONARY.get(word, word)
        # Saring jika bukan stopword selektif
        if norm_word not in STOPWORDS_SELECTIVE and len(norm_word) > 1:
            normalized_words.append(norm_word)

    cleaned_text = " ".join(normalized_words).strip()
    return cleaned_text
