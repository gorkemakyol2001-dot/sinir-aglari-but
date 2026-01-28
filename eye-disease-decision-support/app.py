"""
Göz Hastalığı Karar Destek Sistemi - Flask Web Uygulaması
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import config

# Flask uygulaması
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE
app.config['UPLOAD_FOLDER'] = config.UPLOAD_DIR

# Model yükleme
model = None


def load_trained_model():
    """
    Eğitilmiş modeli yükler.
    """
    global model
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ HATA: Model dosyası bulunamadı: {config.MODEL_PATH}")
        print("⚠️  Lütfen önce 'python train.py' komutunu çalıştırarak modeli eğitin.")
        return False
    
    try:
        model = load_model(config.MODEL_PATH)
        print(f"✅ Model başarıyla yüklendi: {config.MODEL_PATH}")
        
        # Model warmup (ilk tahmin için)
        dummy_input = np.random.rand(1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        _ = model.predict(dummy_input, verbose=0)
        print("✅ Model warmup tamamlandı")
        
        return True
    except Exception as e:
        print(f"❌ Model yükleme hatası: {str(e)}")
        return False


def allowed_file(filename):
    """
    Dosya uzantısını kontrol eder.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Görüntüyü model için hazırlar.
    """
    img = load_img(image_path, target_size=config.IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


@app.route('/')
def index():
    """
    Ana sayfa.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Görüntü tahmin endpoint'i.
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model yüklenmedi. Lütfen önce modeli eğitin.'
        }), 500
    
    # Dosya kontrolü
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Dosya bulunamadı'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Dosya seçilmedi'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Geçersiz dosya formatı. Sadece PNG, JPG, JPEG desteklenir.'
        }), 400
    
    try:
        # Dosyayı kaydet
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Görüntüyü işle
        img_array = preprocess_image(filepath)
        
        # Tahmin yap
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Sınıf indekslerini al (train.py'deki sırayla aynı olmalı)
        class_indices = {i: name for i, name in enumerate(config.CLASS_NAMES_EN)}
        
        # En yüksek 3 tahmini al
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_3_indices:
            class_name_en = class_indices[idx]
            class_name_tr = config.CLASS_NAMES_TR.get(class_name_en, class_name_en)
            confidence = float(predictions[idx]) * 100
            
            results.append({
                'disease': class_name_tr,
                'confidence': round(confidence, 2),
                'description': config.DISEASE_INFO.get(class_name_tr, '')
            })
        
        # Tahmin edilen sınıf
        predicted_idx = top_3_indices[0]
        predicted_class_en = class_indices[predicted_idx]
        predicted_class_tr = config.CLASS_NAMES_TR.get(predicted_class_en, predicted_class_en)
        predicted_confidence = float(predictions[predicted_idx]) * 100
        
        # Yüklenen dosyanın URL'si
        file_url = f'/static/uploads/{filename}'
        
        return jsonify({
            'success': True,
            'prediction': {
                'disease': predicted_class_tr,
                'confidence': round(predicted_confidence, 2),
                'description': config.DISEASE_INFO.get(predicted_class_tr, '')
            },
            'top_predictions': results,
            'image_url': file_url
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Tahmin hatası: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """
    Sağlık kontrolü endpoint'i.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("GÖZ HASTALIĞI KARAR DESTEK SİSTEMİ")
    print("WEB UYGULAMASI")
    print("=" * 60)
    
    # Upload klasörünü oluştur
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    
    # Modeli yükle
    if load_trained_model():
        print(f"\n🌐 Web uygulaması başlatılıyor...")
        print(f"📍 Adres: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
        print(f"🔗 Tarayıcınızda açın: http://localhost:{config.FLASK_PORT}")
        print("\n⚠️  Uygulamayı durdurmak için: CTRL+C\n")
        
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG
        )
    else:
        print("\n❌ Model yüklenemedi. Uygulama başlatılamıyor.")
        print("💡 Çözüm: Önce 'python train.py' komutunu çalıştırın.")
