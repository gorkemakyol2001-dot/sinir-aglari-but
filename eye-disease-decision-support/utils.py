"""
Göz Hastalığı Karar Destek Sistemi - Yardımcı Fonksiyonlar
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import config


def create_data_splits():
    """
    Augmented Dataset'ten train/validation/test ayrımı yapar.
    """
    print("=" * 60)
    print("VERİ SETİ AYIRMA İŞLEMİ BAŞLIYOR")
    print("=" * 60)
    
    # Hedef klasörleri temizle ve oluştur
    for split_dir in [config.TRAIN_DIR, config.VALIDATION_DIR, config.TEST_DIR]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
    
    # Her sınıf için işlem yap
    for class_name in config.CLASS_NAMES_EN:
        source_dir = os.path.join(config.AUGMENTED_DATASET, class_name)
        
        if not os.path.exists(source_dir):
            print(f"⚠️  UYARI: {class_name} klasörü bulunamadı!")
            continue
        
        # Görüntüleri al
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            print(f"⚠️  UYARI: {class_name} klasöründe görüntü bulunamadı!")
            continue
        
        # Train/temp split (70% train, 30% temp)
        train_images, temp_images = train_test_split(
            images, 
            test_size=(1 - config.TRAIN_SPLIT),
            random_state=config.RANDOM_SEED
        )
        
        # Validation/test split (15% validation, 15% test)
        val_images, test_images = train_test_split(
            temp_images,
            test_size=config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT),
            random_state=config.RANDOM_SEED
        )
        
        # Klasörleri oluştur
        for split_name in ['train', 'validation', 'test']:
            split_class_dir = os.path.join(config.DATA_DIR, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
        
        # Dosyaları kopyala
        for img in train_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(config.TRAIN_DIR, class_name, img)
            shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(config.VALIDATION_DIR, class_name, img)
            shutil.copy2(src, dst)
        
        for img in test_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(config.TEST_DIR, class_name, img)
            shutil.copy2(src, dst)
        
        print(f"✅ {class_name}:")
        print(f"   Train: {len(train_images)} | Validation: {len(val_images)} | Test: {len(test_images)}")
    
    print("\n" + "=" * 60)
    print("VERİ SETİ AYIRMA İŞLEMİ TAMAMLANDI")
    print("=" * 60)
    
    # Özet istatistikler
    print_data_summary()


def print_data_summary():
    """
    Veri seti hakkında özet bilgi verir.
    """
    print("\n📊 VERİ SETİ ÖZETİ:")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in config.CLASS_NAMES_EN:
        train_path = os.path.join(config.TRAIN_DIR, class_name)
        val_path = os.path.join(config.VALIDATION_DIR, class_name)
        test_path = os.path.join(config.TEST_DIR, class_name)
        
        train_count = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        val_count = len(os.listdir(val_path)) if os.path.exists(val_path) else 0
        test_count = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print(f"Toplam Eğitim Görüntüsü: {total_train}")
    print(f"Toplam Doğrulama Görüntüsü: {total_val}")
    print(f"Toplam Test Görüntüsü: {total_test}")
    print(f"TOPLAM: {total_train + total_val + total_test}")
    print("-" * 60)


def create_data_generators():
    """
    Data augmentation ile veri jeneratörleri oluşturur.
    """
    # Training data generator (augmentation ile)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.AUGMENTATION_PARAMS['rotation_range'],
        width_shift_range=config.AUGMENTATION_PARAMS['width_shift_range'],
        height_shift_range=config.AUGMENTATION_PARAMS['height_shift_range'],
        horizontal_flip=config.AUGMENTATION_PARAMS['horizontal_flip'],
        zoom_range=config.AUGMENTATION_PARAMS['zoom_range'],
        brightness_range=config.AUGMENTATION_PARAMS['brightness_range'],
        fill_mode=config.AUGMENTATION_PARAMS['fill_mode']
    )
    
    # Validation ve test data generator (sadece rescale)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generators oluştur
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n✅ Data generators oluşturuldu!")
    print(f"Sınıf sayısı: {len(train_generator.class_indices)}")
    print(f"Sınıflar: {list(train_generator.class_indices.keys())}")
    
    return train_generator, validation_generator, test_generator


def load_and_preprocess_image(image_path):
    """
    Tek bir görüntüyü yükler ve model için hazırlar.
    
    Args:
        image_path: Görüntü dosyasının yolu
        
    Returns:
        Preprocessed görüntü array (1, 224, 224, 3)
    """
    img = load_img(image_path, target_size=config.IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


def plot_sample_images(num_samples=5):
    """
    Her sınıftan örnek görüntüler gösterir.
    
    Args:
        num_samples: Her sınıftan kaç örnek gösterileceği
    """
    num_classes = len(config.CLASS_NAMES_EN)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 3 * num_classes))
    fig.suptitle('Veri Setinden Örnek Görüntüler', fontsize=16, fontweight='bold')
    
    for i, class_name in enumerate(config.CLASS_NAMES_EN):
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Rastgele örnekler seç
        samples = np.random.choice(images, min(num_samples, len(images)), replace=False)
        
        for j, img_name in enumerate(samples):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=config.IMG_SIZE)
            
            if num_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            ax.imshow(img)
            ax.axis('off')
            
            if j == 0:
                # Türkçe sınıf adını göster
                tr_name = config.CLASS_NAMES_TR.get(class_name, class_name)
                ax.set_ylabel(tr_name, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Örnek görüntüler kaydedildi: {save_path}")
    plt.close()


def plot_class_distribution():
    """
    Sınıf dağılımını görselleştirir.
    """
    train_counts = []
    val_counts = []
    test_counts = []
    class_labels = []
    
    for class_name in config.CLASS_NAMES_EN:
        train_path = os.path.join(config.TRAIN_DIR, class_name)
        val_path = os.path.join(config.VALIDATION_DIR, class_name)
        test_path = os.path.join(config.TEST_DIR, class_name)
        
        train_count = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        val_count = len(os.listdir(val_path)) if os.path.exists(val_path) else 0
        test_count = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
        
        train_counts.append(train_count)
        val_counts.append(val_count)
        test_counts.append(test_count)
        
        # Türkçe sınıf adı
        tr_name = config.CLASS_NAMES_TR.get(class_name, class_name)
        class_labels.append(tr_name)
    
    # Grafik oluştur
    x = np.arange(len(class_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, train_counts, width, label='Eğitim', color='#3498db')
    bars2 = ax.bar(x, val_counts, width, label='Doğrulama', color='#2ecc71')
    bars3 = ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c')
    
    ax.set_xlabel('Hastalık Sınıfları', fontsize=12, fontweight='bold')
    ax.set_ylabel('Görüntü Sayısı', fontsize=12, fontweight='bold')
    ax.set_title('Veri Seti Dağılımı', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Bar üzerine değerleri yaz
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sınıf dağılımı grafiği kaydedildi: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test için
    print("Utils modülü yüklendi.")
    print(f"Veri seti yolu: {config.AUGMENTED_DATASET}")
    print(f"Sınıf sayısı: {len(config.CLASS_NAMES_EN)}")
