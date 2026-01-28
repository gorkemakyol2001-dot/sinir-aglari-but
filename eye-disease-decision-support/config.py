"""
Göz Hastalığı Karar Destek Sistemi - Konfigürasyon Dosyası
"""

import os

# Proje dizinleri
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')

# Veri seti dizinleri
ORIGINAL_DATASET = r'c:\Users\Lenovo\Desktop\sinir-aglari\Original Dataset'
AUGMENTED_DATASET = r'c:\Users\Lenovo\Desktop\sinir-aglari\Augmented Dataset'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model parametreleri
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Sınıf isimleri (İngilizce - klasör adları)
CLASS_NAMES_EN = [
    'Central Serous Chorioretinopathy [Color Fundus]',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

# Sınıf isimleri (Türkçe - web arayüzü için)
CLASS_NAMES_TR = {
    'Central Serous Chorioretinopathy [Color Fundus]': 'Santral Seröz Korioretinopati',
    'Diabetic Retinopathy': 'Diyabetik Retinopati',
    'Disc Edema': 'Disk Ödemesi',
    'Glaucoma': 'Glokom',
    'Healthy': 'Sağlıklı Göz',
    'Macular Scar': 'Maküler Skar',
    'Myopia': 'Miyopi',
    'Pterygium': 'Pterjium',
    'Retinal Detachment': 'Retina Dekolmanı',
    'Retinitis Pigmentosa': 'Retinitis Pigmentosa'
}

# Hastalık açıklamaları
DISEASE_INFO = {
    'Santral Seröz Korioretinopati': 'Retina altında sıvı birikmesi ile karakterize bir göz hastalığıdır.',
    'Diyabetik Retinopati': 'Diyabet hastalığının retinada neden olduğu hasar ve görme kaybıdır.',
    'Disk Ödemesi': 'Optik sinir başının şişmesi ile karakterize bir durumdur.',
    'Glokom': 'Göz içi basıncının artması sonucu optik sinir hasarı ve görme kaybıdır.',
    'Sağlıklı Göz': 'Herhangi bir patolojik bulgu içermeyen normal göz fundus görüntüsü.',
    'Maküler Skar': 'Makula bölgesinde oluşan skar dokusu, merkezi görmeyi etkiler.',
    'Miyopi': 'Yakını net, uzağı bulanık görme durumu (yakın görüşlülük).',
    'Pterjium': 'Konjonktivadan korneaya doğru büyüyen anormal doku.',
    'Retina Dekolmanı': 'Retinanın alttaki dokulardan ayrılması, acil müdahale gerektirir.',
    'Retinitis Pigmentosa': 'Genetik bir göz hastalığı, retinada pigment birikimi ve görme kaybı.'
}

# Model dosya yolu
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

# Data Augmentation parametreleri
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Callback parametreleri
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Flask ayarları
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Veri bölme oranları
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed (tekrarlanabilirlik için)
RANDOM_SEED = 42
