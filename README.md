# 🏥 Göz Hastalığı Karar Destek Sistemi

> **CNN-based eye disease decision support system with dataset**

Fundus kamera görüntülerinden yapay zeka destekli göz hastalığı tespiti yapan derin öğrenme tabanlı karar destek sistemi.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)



### 💻 Kullanılan Diller
- **Python** 83.3%
- **CSS** 11.2%
- **HTML** 1.7%
- **JavaScript** 3.8%

## 🌟 Öne Çıkan Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🧠 **Yapay Zeka** | EfficientNetB0 transfer learning ile %85+ doğruluk |
| 🖼️ **Kolay Kullanım** | Drag & drop ile görüntü yükleme |
| ⚡ **Hızlı Sonuç** | Saniyeler içinde tahmin |
| 📊 **Detaylı Analiz** | Güven skorları ve alternatif tanılar |
| 🎨 **Modern Tasarım** | Responsive ve kullanıcı dostu arayüz |
| 🔒 **Güvenli** | Yerel işleme, veri gizliliği |

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)

## 🎯 Proje Hakkında

Bu proje, fundus kamera görüntülerini analiz ederek 10 farklı göz hastalığını tespit edebilen bir derin öğrenme sistemidir. Transfer learning tekniği kullanılarak EfficientNetB0 mimarisi ile eğitilmiştir.

### Tespit Edilebilen Hastalıklar

1. **Santral Seröz Korioretinopati** - Retina altında sıvı birikmesi
2. **Diyabetik Retinopati** - Diyabetin neden olduğu retina hasarı
3. **Disk Ödemesi** - Optik sinir başının şişmesi
4. **Glokom** - Göz içi basıncı artışı ve optik sinir hasarı
5. **Sağlıklı Göz** - Normal fundus görüntüsü
6. **Maküler Skar** - Makula bölgesinde skar dokusu
7. **Miyopi** - Yakın görüşlülük
8. **Pterjium** - Konjonktivadan korneaya büyüyen doku
9. **Retina Dekolmanı** - Retinanın ayrılması
10. **Retinitis Pigmentosa** - Genetik retina hastalığı

## ✨ Özellikler

- ✅ **Transfer Learning** ile EfficientNetB0 mimarisi
- ✅ **Data Augmentation** ile güçlendirilmiş eğitim
- ✅ **Modern Web Arayüzü** - Drag & drop görüntü yükleme
- ✅ **Gerçek Zamanlı Tahmin** - Hızlı ve doğru sonuçlar
- ✅ **Detaylı Raporlama** - Confusion matrix, ROC eğrileri
- ✅ **Responsive Tasarım** - Tüm cihazlarda çalışır

## 📊 Veri Seti

Proje, fundus kamera görüntülerinden oluşan augmented dataset kullanılarak eğitilmiştir.

**Veri Bölünmesi:**
- Eğitim (Train): %70
- Doğrulama (Validation): %15
- Test: %15

**Veri Artırma Teknikleri:**
- Rotasyon (±20°)
- Kaydırma (±20%)
- Yatay çevirme
- Zoom (±20%)
- Parlaklık ayarı (±20%)

## 🏗️ Model Mimarisi

### Base Model: EfficientNetB0
- **Önceden Eğitilmiş Ağırlıklar:** ImageNet
- **Input Shape:** (224, 224, 3)
- **Pooling:** Global Average Pooling

### Custom Layers
```
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(10, Softmax)
```

### Hiperparametreler
- **Optimizer:** Adam (lr=0.0001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 50 (Early Stopping ile)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## 🚀 Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adım 1: Repository'yi Klonlayın
```bash
git clone https://github.com/[kullanıcı-adınız]/sinir-aglari-eye-disease-decision-support.git
cd eye-disease-decision-support
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Veri Setini Hazırlayın
Veri setinizi `config.py` dosyasında belirtilen konuma yerleştirin.

## 💻 Kullanım

### Model Eğitimi
```bash
python train.py
```

Bu komut:
- Veri setini train/validation/test olarak böler
- Data augmentation uygular
- Modeli eğitir
- Sonuçları `results/` klasörüne kaydeder
- En iyi modeli `models/best_model.h5` olarak kaydeder

### Web Uygulamasını Çalıştırma
```bash
python app.py
```

Tarayıcınızda `http://localhost:5000` adresini açın.

### Web Arayüzü Kullanımı
1. Fundus görüntüsünü yükleyin (drag & drop veya tıklayarak)
2. "Tahmin Et" butonuna tıklayın
3. Sonuçları görüntüleyin:
   - Tespit edilen hastalık
   - Güven skoru (%)
   - En olası 3 tahmin
   - Hastalık açıklaması

## 📁 Proje Yapısı

```
eye-disease-decision-support/
│
├── data/                          # Veri seti klasörü
│   ├── train/                     # Eğitim verisi
│   ├── validation/                # Doğrulama verisi
│   └── test/                      # Test verisi
│
├── models/                        # Eğitilmiş modeller
│   └── best_model.h5             # En iyi model
│
├── results/                       # Eğitim sonuçları
│   ├── confusion_matrix.png      # Confusion matrix
│   ├── training_history.png      # Eğitim grafikleri
│   ├── roc_curves.png            # ROC eğrileri
│   ├── class_distribution.png    # Sınıf dağılımı
│   └── classification_report.txt # Detaylı rapor
│
├── static/                        # Web statik dosyalar
│   ├── css/
│   │   └── style.css             # Modern CSS
│   ├── js/
│   │   └── main.js               # JavaScript
│   └── uploads/                   # Yüklenen görüntüler
│
├── templates/                     # Flask HTML şablonları
│   └── index.html                # Ana sayfa
│
├── app.py                         # Flask web uygulaması
├── train.py                       # Model eğitim scripti
├── utils.py                       # Yardımcı fonksiyonlar
├── config.py                      # Konfigürasyon
├── requirements.txt               # Python bağımlılıkları
├── README.md                      # Bu dosya
└── PROJE_RAPORU.md               # Akademik rapor
```

## 📈 Sonuçlar

Model eğitimi tamamlandıktan sonra `results/` klasöründe aşağıdaki dosyalar oluşturulur





### 📋 Performans Metrikleri

Model, test seti üzerinde aşağıdaki performans metriklerini elde etmiştir:

| Metrik | Değer |
|--------|-------|
| **Accuracy** | %85+ |
| **Precision** | %83+ |
| **Recall** | %82+ |
| **F1-Score** | %82+ |

**Detaylı Rapor:** `results/classification_report.txt` dosyasında sınıf bazında detaylı metrikler bulunmaktadır.

## 🛠️ Teknolojiler

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras
- **Model:** EfficientNetB0 (Transfer Learning)
- **Veri İşleme:** NumPy, Pandas, OpenCV
- **Görselleştirme:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript
- **Tasarım:** Modern UI/UX, Glassmorphism

## ⚠️ Önemli Notlar

> **DİKKAT:** Bu sistem bir karar destek aracıdır ve kesin tanı koyamaz. Elde edilen sonuçlar mutlaka bir göz hekimi tarafından değerlendirilmelidir.


## 👥 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

Sorularınız için issue açabilirsiniz.

## 🙏 Teşekkürler

Bu proje, derin öğrenme ve tıbbi görüntü analizi alanındaki araştırmalara katkıda bulunmayı amaçlamaktadır.

---

**© 2026 Göz Hastalığı Karar Destek Sistemi | Derin Öğrenme Projesi**
