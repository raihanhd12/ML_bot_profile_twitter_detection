# 🤖 Twitter Bot Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)

Sistem deteksi bot Twitter yang menggunakan machine learning untuk membedakan akun bot dan manusia berdasarkan analisis fitur profil dan perilaku pengguna.

## 📋 Daftar Isi

- [Gambaran Proyek](#-gambaran-proyek)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#️-instalasi)
- [Penggunaan](#-penggunaan)
- [Dataset](#-dataset)
- [Model & Performa](#-model--performa)
- [Fitur-fitur Utama](#-fitur-fitur-utama)
- [Kontribusi](#-kontribusi)

## 🎯 Gambaran Proyek

Proyek ini mengimplementasikan sistem deteksi bot otomatis yang menganalisis karakteristik akun Twitter untuk membedakan antara akun manusia asli dan akun bot. Model ini mencapai akurasi tinggi dengan memeriksa berbagai pola perilaku dan fitur profil.

### ✨ Fitur Utama

- **Analisis Data Eksploratori (EDA)** yang komprehensif
- **Feature Engineering** untuk ekstraksi fitur behavioral dan profil
- **Multiple ML Models** dengan perbandingan performa
- **Visualisasi** data dan hasil analisis
- **Model Evaluation** dengan cross-validation

## 📁 Struktur Proyek

```
bot-detection-twitter/
├── 0_dataset_ai.ipynb          # Eksplorasi dataset awal
├── 1_model_inspect.ipynb       # Inspeksi dan analisis model
├── 2_dev.ipynb                 # Development dan eksperimen model
├── requirements.txt            # Dependencies Python
├── data/                       # Directory dataset
├── .gitignore                  # Git ignore rules
└── README.md                   # Dokumentasi proyek
```

## 🛠️ Instalasi

### Persyaratan Sistem

- Python 3.8 atau lebih tinggi
- pip atau conda package manager
- Jupyter Notebook

### Langkah Instalasi

1. **Clone repository**

```bash
git clone https://github.com/username/bot-detection-twitter.git
cd bot-detection-twitter
```

2. **Buat virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Di Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Jalankan Jupyter Notebook**

```bash
jupyter notebook
```

## 🚀 Penggunaan

### Menjalankan Analisis

1. **Eksplorasi Dataset**

   - Buka `0_dataset_ai.ipynb` untuk melihat analisis dataset awal

2. **Development & Training Model**

   - Jalankan `2_dev.ipynb` untuk training dan eksperimen model

3. **Inspeksi Model**
   - Gunakan `1_model_inspect.ipynb` untuk analisis mendalam model

### Contoh Penggunaan Model

```python
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Contoh data akun Twitter
sample_data = {
    'favourites_count': 1000,
    'followers_count': 500,
    'friends_count': 200,
    'statuses_count': 1500,
    'account_age_days': 365,
    'follower_following_ratio': 2.5,
    'bio_length': 120,
    'username_length': 12,
    'has_custom_profile_image': 1,
    'is_verified': 0
}

# Prediksi (setelah model dilatih dan disimpan)
# prediction = model.predict([list(sample_data.values())])
# print(f"Prediksi: {'Bot' if prediction[0] == 1 else 'Human'}")
```

## � Dataset

Proyek ini menggunakan dataset **Twitter Bot Detection** yang berisi:

- **Ukuran**: Ribuan akun Twitter
- **Fitur**: 20+ karakteristik akun
- **Label**: Binary classification (Human/Bot)
- **Sumber**: Data akun Twitter asli

### Fitur-fitur Utama

#### Fitur Numerik

- `favourites_count`: Jumlah tweet yang disukai
- `followers_count`: Jumlah followers
- `friends_count`: Jumlah akun yang diikuti
- `statuses_count`: Total tweet yang diposting
- `account_age_days`: Umur akun dalam hari
- `follower_following_ratio`: Rasio followers to following
- `bio_length`: Panjang bio profil
- `username_length`: Panjang username

#### Fitur Binary

- `has_custom_profile_image`: Foto profil custom
- `has_location`: Informasi lokasi tersedia
- `is_verified`: Status verifikasi akun
- `is_geo_enabled`: Geo-location diaktifkan

## 📈 Model & Performa

### Model yang Digunakan

- **Random Forest**: Model ensemble berbasis pohon keputusan
- **XGBoost**: Gradient boosting yang dioptimasi
- **LightGBM**: Gradient boosting yang efisien
- **Logistic Regression**: Model linear untuk baseline
- **Support Vector Machine**: Model dengan kernel RBF

### Evaluasi Model

Model dievaluasi menggunakan:

- **Cross-validation** dengan stratified k-fold
- **Confusion Matrix** untuk analisis klasifikasi
- **ROC-AUC Score** untuk performa keseluruhan
- **Precision, Recall, F1-Score** untuk metrik detail

### Performa Terbaik

| Metrik    | Skor  |
| --------- | ----- |
| Accuracy  | ~95%+ |
| Precision | ~94%+ |
| Recall    | ~93%+ |
| F1-Score  | ~94%+ |
| ROC-AUC   | ~97%+ |

## 🔍 Fitur-fitur Utama

### Analisis Data Eksploratori

- Distribusi fitur untuk bot vs human
- Analisis korelasi antar fitur
- Statistik deskriptif untuk setiap grup
- Visualisasi perbandingan karakteristik

### Feature Engineering

- Ekstraksi fitur behavioral dari data profil
- Normalisasi dan scaling fitur numerik
- Encoding fitur kategorikal
- Feature selection berdasarkan importance

### Insights Utama

**Karakteristik Bot:**

- Rasio follower-to-following yang tinggi
- Bio yang pendek atau generic
- Username dengan banyak angka
- Profil kurang terkustomisasi
- Akun yang relatif baru

**Karakteristik Human:**

- Rasio follower yang lebih seimbang
- Profil yang dipersonalisasi
- Pola aktivitas yang bervariasi
- Riwayat akun yang lebih panjang

## � Pengembangan Selanjutnya

- [ ] **Analisis Teks**: Analisis konten tweet
- [ ] **Network Analysis**: Fitur jaringan sosial
- [ ] **Real-time Detection**: API untuk deteksi live
- [ ] **Model Ensemble**: Kombinasi multiple models
- [ ] **Deployment**: Web application untuk demo

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat branch fitur (`git checkout -b fitur/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Tambah AmazingFeature'`)
4. Push ke branch (`git push origin fitur/AmazingFeature`)
5. Buat Pull Request

## � Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## �📞 Kontak

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your-email@example.com

## 🙏 Acknowledgments

- Dataset dari Hugging Face community
- Scikit-learn, XGBoost, LightGBM libraries
- Komunitas machine learning Indonesia

---

⭐ **Jika proyek ini membantu, berikan star!** ⭐

---

**Terakhir diupdate**: Juli 2025
