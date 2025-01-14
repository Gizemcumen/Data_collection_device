from flask import Flask, jsonify, request, send_file
from .camera import Camera
import os
from datetime import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)
camera = Camera()

# Modeli yükle
model = None
try:
    model = tf.keras.models.load_model('saved_models/wall_quality_model.h5')
except:
    print("Model henüz yüklenmedi")


@app.route('/capture', methods=['POST'])
def capture():
    try:
        os.makedirs("captured_images", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"captured_images/wall_{timestamp}.jpg"

        camera.capture(image_path)

        return jsonify({
            'status': 'success',
            'image_path': image_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model yüklenmedi'}), 500

        if 'image' not in request.files:
            return jsonify({'error': 'Görüntü bulunamadı'}), 400

        image_file = request.files['image']
        image = tf.keras.preprocessing.image.load_img(
            image_file, target_size=(224, 224)
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        prediction = model.predict(image_array)
        result = "İyi" if prediction[0][0] > 0.5 else "Kötü"

        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': float(prediction[0][0])
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
afrom flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from camera import Camera
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing'i etkinleştir

# Kamera nesnesini oluştur
camera = None
try:
    camera = Camera()
except Exception as e:
    print(f"Kamera başlatılamadı: {str(e)}")

# Model yükleme
model = None
try:
    model = tf.keras.models.load_model('../model/saved_models/wall_quality_model.h5')
    print("Model başarıyla yüklendi")
except Exception as e:
    print(f"Model yüklenemedi: {str(e)}")

def ensure_dir(directory):
    """Klasörün var olduğundan emin ol"""
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.route('/health', methods=['GET'])
def health_check():
    """Sunucu durumunu kontrol et"""
    status = {
        'server': 'running',
        'camera': camera is not None,
        'model': model is not None
    }
    return jsonify(status)

@app.route('/capture', methods=['POST'])
def capture():
    """Fotoğraf çek"""
    if not camera:
        return jsonify({
            'status': 'error',
            'message': 'Kamera başlatılamadı'
        }), 500

    try:
        # Captured images klasörünü kontrol et
        ensure_dir('captured_images')

        # Fotoğraf çek
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"captured_images/wall_{timestamp}.jpg"
        camera.capture(image_path)

        # Dosyanın var olduğunu kontrol et
        if not os.path.exists(image_path):
            raise Exception("Fotoğraf kaydedilemedi")

        return jsonify({
            'status': 'success',
            'image_path': image_path,
            'timestamp': timestamp
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/image/<path:filename>')
def serve_image(filename):
    """Çekilen fotoğrafı gönder"""
    try:
        return send_file(f"captured_images/{filename}")
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Görüntü bulunamadı: {str(e)}"
        }), 404

@app.route('/predict', methods=['POST'])
def predict():
    """Fotoğraf analizi yap"""
    if not model:
        return jsonify({
            'status': 'error',
            'message': 'Model yüklenmedi'
        }), 500

    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'Görüntü bulunamadı'
        }), 400

    try:
        # Görüntüyü oku ve ön işle
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Tahmin yap
        prediction = model.predict(image_array)
        probability = float(prediction[0][0])
        result = "İyi" if probability > 0.5 else "Kötü"
        confidence = probability if result == "İyi" else 1 - probability

        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': float(confidence * 100),
            'raw_probability': float(probability)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Tahmin hatası: {str(e)}"
        }), 500

@app.route('/list_images', methods=['GET'])
def list_images():
    """Çekilen fotoğrafları listele"""
    try:
        images_dir = 'captured_images'
        ensure_dir(images_dir)

        images = []
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, filename)
                images.append({
                    'filename': filename,
                    'path': image_path,
                    'timestamp': os.path.getctime(image_path)
                })

        return jsonify({
            'status': 'success',
            'images': sorted(images, key=lambda x: x['timestamp'], reverse=True)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Uygulama başlangıcında gerekli klasörleri oluştur
        ensure_dir('captured_images')

        # Sunucuyu başlat
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Sunucu başlatılamadı: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)