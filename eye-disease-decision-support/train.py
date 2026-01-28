"""
Göz Hastalığı Karar Destek Sistemi - Model Eğitim Scripti
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    CSVLogger
)
from tensorflow.keras.metrics import Precision, Recall, AUC

from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import config
from utils import (
    create_data_splits,
    create_data_generators,
    plot_sample_images,
    plot_class_distribution
)


def build_model():
    """
    EfficientNetB0 tabanlı transfer learning modeli oluşturur.
    """
    print("\n" + "=" * 60)
    print("MODEL MİMARİSİ OLUŞTURULUYOR")
    print("=" * 60)
    
    # Base model (EfficientNetB0)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        pooling='avg'
    )
    
    # Base model katmanlarını dondur
    base_model.trainable = False
    
    # Custom layers ekle
    x = base_model.output
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    outputs = Dense(len(config.CLASS_NAMES_EN), activation='softmax', name='predictions')(x)
    
    # Final model
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Model compile
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    
    # Model özeti
    print("\n📊 MODEL ÖZETİ:")
    print("-" * 60)
    model.summary()
    
    # Parametre sayıları
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    
    print("\n📈 PARAMETRE İSTATİSTİKLERİ:")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    print(f"Eğitilemez parametreler: {non_trainable_params:,}")
    print(f"Toplam parametreler: {trainable_params + non_trainable_params:,}")
    print("-" * 60)
    
    return model


def get_callbacks():
    """
    Eğitim için callback fonksiyonlarını oluşturur.
    """
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce Learning Rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        
        # Model Checkpoint
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # CSV Logger
        CSVLogger(
            filename=os.path.join(config.RESULTS_DIR, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    return callbacks


def train_model(model, train_gen, val_gen):
    """
    Modeli eğitir.
    """
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Callbacks
    callbacks = get_callbacks()
    
    # Eğitim
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ TAMAMLANDI")
    print(f"Eğitim Süresi: {training_time}")
    print("=" * 60)
    
    return history


def plot_training_history(history):
    """
    Eğitim geçmişini görselleştirir.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Eğitim Sonuçları', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Eğitim', linewidth=2, color='#3498db')
    axes[0, 0].plot(history.history['val_accuracy'], label='Doğrulama', linewidth=2, color='#e74c3c')
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Eğitim', linewidth=2, color='#3498db')
    axes[0, 1].plot(history.history['val_loss'], label='Doğrulama', linewidth=2, color='#e74c3c')
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Eğitim', linewidth=2, color='#3498db')
    axes[1, 0].plot(history.history['val_precision'], label='Doğrulama', linewidth=2, color='#e74c3c')
    axes[1, 0].set_title('Model Precision', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Eğitim', linewidth=2, color='#3498db')
    axes[1, 1].plot(history.history['val_recall'], label='Doğrulama', linewidth=2, color='#e74c3c')
    axes[1, 1].set_title('Model Recall', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Eğitim grafikleri kaydedildi: {save_path}")
    plt.close()


def evaluate_model(model, test_gen):
    """
    Modeli test seti üzerinde değerlendirir.
    """
    print("\n" + "=" * 60)
    print("MODEL DEĞERLENDİRME")
    print("=" * 60)
    
    # Test seti üzerinde tahmin
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Gerçek sınıflar
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    # Türkçe sınıf isimleri
    class_labels_tr = [config.CLASS_NAMES_TR.get(label, label) for label in class_labels]
    
    # Classification Report
    print("\n📊 CLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_labels_tr,
        digits=4
    )
    print(report)
    
    # Raporu dosyaya kaydet
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GÖZ HASTALIĞI KARAR DESTEK SİSTEMİ\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"✅ Classification report kaydedildi: {report_path}")
    
    # Confusion Matrix
    generate_confusion_matrix(true_classes, predicted_classes, class_labels_tr)
    
    # ROC Curves
    plot_roc_curves(true_classes, predictions, class_labels_tr)
    
    # Test accuracy
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_gen, verbose=0)
    print(f"\n📈 TEST SETİ PERFORMANSI:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    return predictions, predicted_classes, true_classes


def generate_confusion_matrix(true_classes, predicted_classes, class_labels):
    """
    Confusion matrix oluşturur ve görselleştirir.
    """
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Ham confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axes[0],
        cbar_kws={'label': 'Sayı'}
    )
    axes[0].set_title('Confusion Matrix (Ham Değerler)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Tahmin Edilen', fontweight='bold')
    axes[0].set_ylabel('Gerçek', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalize confusion matrix
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Greens',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axes[1],
        cbar_kws={'label': 'Oran'}
    )
    axes[1].set_title('Confusion Matrix (Normalize)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Tahmin Edilen', fontweight='bold')
    axes[1].set_ylabel('Gerçek', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix kaydedildi: {save_path}")
    plt.close()


def plot_roc_curves(true_classes, predictions, class_labels):
    """
    Her sınıf için ROC eğrisi çizer.
    """
    n_classes = len(class_labels)
    
    # One-hot encode true classes
    true_classes_bin = label_binarize(true_classes, classes=range(n_classes))
    
    # ROC curve ve AUC hesapla
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_classes_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i], 
            tpr[i], 
            color=color, 
            lw=2,
            label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele Tahmin')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Eğrileri (Tüm Sınıflar)', fontweight='bold', fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ ROC eğrileri kaydedildi: {save_path}")
    plt.close()


def main():
    """
    Ana eğitim pipeline'ı.
    """
    print("\n" + "=" * 60)
    print("GÖZ HASTALIĞI KARAR DESTEK SİSTEMİ")
    print("MODEL EĞİTİM PİPELINE")
    print("=" * 60)
    
    # GPU kontrolü
    print(f"\n🖥️  GPU Durumu: {tf.config.list_physical_devices('GPU')}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 1. Veri setini ayır
    print("\n📁 ADIM 1: Veri Seti Hazırlama")
    create_data_splits()
    
    # 2. Veri görselleştirme
    print("\n📊 ADIM 2: Veri Görselleştirme")
    plot_class_distribution()
    plot_sample_images(num_samples=5)
    
    # 3. Data generators oluştur
    print("\n🔄 ADIM 3: Data Generators Oluşturma")
    train_gen, val_gen, test_gen = create_data_generators()
    
    # 4. Model oluştur
    print("\n🏗️  ADIM 4: Model Mimarisi Oluşturma")
    model = build_model()
    
    # 5. Model eğit
    print("\n🚀 ADIM 5: Model Eğitimi")
    history = train_model(model, train_gen, val_gen)
    
    # 6. Eğitim sonuçlarını görselleştir
    print("\n📈 ADIM 6: Eğitim Sonuçlarını Görselleştirme")
    plot_training_history(history)
    
    # 7. Modeli değerlendir
    print("\n🎯 ADIM 7: Model Değerlendirme")
    predictions, predicted_classes, true_classes = evaluate_model(model, test_gen)
    
    print("\n" + "=" * 60)
    print("✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    print(f"\n📁 Model kaydedildi: {config.MODEL_PATH}")
    print(f"📁 Sonuçlar klasörü: {config.RESULTS_DIR}")
    print("\n🌐 Web uygulamasını çalıştırmak için: python app.py")


if __name__ == "__main__":
    main()
