from src.preprocessing import clean_text

def test_clean_text_basic():
    raw = "Barang bagus bgt, tapi pengirimannya lemot bgt!! Gak suka."
    cleaned = clean_text(raw)
    assert "bagus" in cleaned
    assert "banget" in cleaned
    assert "tidak" in cleaned
    assert "suka" in cleaned
    assert "!!" not in cleaned

def test_clean_text_slang_and_repeats():
    raw = "Aplikasinya mantullll bgttt, makasih tokped! Semoga makin top!!"
    cleaned = clean_text(raw)
    assert "aplikasi" in cleaned
    assert "bagus" in cleaned
    assert "banget" in cleaned
    assert "terima kasih" in cleaned
    assert "tokopedia" in cleaned

def test_clean_text_negation_preserved():
    raw = "Saya tidak puas karena barang belum sampai dan rusak parah!"
    cleaned = clean_text(raw)
    assert "tidak" in cleaned
    assert "belum" in cleaned
    assert "rusak" in cleaned

def test_clean_text_empty_and_symbols():
    assert clean_text("") == ""
    assert clean_text("12345 !@#$%^&*()") == ""
