import sys
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QFrame, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import tensorflow as tf
import numpy as np
import cv2
import os



class PredictionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        print("Initializing PredictionPanel")
        self.model = None
        self.current_image = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        """UI bileşenlerini oluştur"""
        print("Setting up UI components")
        layout = QVBoxLayout()

        # Sonuç başlığı
        self.result_title = QLabel("Tahmin Sonucu")
        self.result_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.result_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_title)

        # Görüntü önizleme
        print("Creating image preview widget")
        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setStyleSheet("border: 2px solid gray")
        self.image_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_preview)

        # UI component creation logs
        print("Creating progress bars")
        prediction_layout = QHBoxLayout()

        # İyi durum göstergesi
        self.good_progress = QProgressBar()
        self.good_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

        # Kötü durum göstergesi
        self.bad_progress = QProgressBar()
        self.bad_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #f44336;
            }
        """)

        prediction_layout.addWidget(QLabel("İyi:"))
        prediction_layout.addWidget(self.good_progress)
        prediction_layout.addWidget(QLabel("Kötü:"))
        prediction_layout.addWidget(self.bad_progress)

        layout.addLayout(prediction_layout)

        print("Setting up result label and feedback buttons")
        # Sonuç metni
        self.result_label = QLabel()
        self.result_label.setFont(QFont('Arial', 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # Feedback butonları
        feedback_layout = QHBoxLayout()

        self.correct_btn = QPushButton("Doğru Tahmin ✓")
        self.correct_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.incorrect_btn = QPushButton("Yanlış Tahmin ✗")
        self.incorrect_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        feedback_layout.addWidget(self.correct_btn)
        feedback_layout.addWidget(self.incorrect_btn)

        layout.addLayout(feedback_layout)

        # Ayırıcı çizgi
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        self.setLayout(layout)
        print("UI setup completed")

    def load_model(self):
        """Eğitilmiş modeli yükle"""
        print("Starting model loading process")
        try:
            # Get the absolute path to the model file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = f"{os.path.dirname(__file__)}/../../model/saved_models/wall_model_final.h5"
            print(f"Model path: {model_path}")

            # Configure model loading to be less strict and handle GPU memory better
            print("Clearing Keras session and configuring GPU")
            tf.keras.backend.clear_session()
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            if gpu_devices:
                print(f"Found GPU devices: {gpu_devices}")
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(f"Enabled memory growth for device: {device}")

            # Load model with custom objects in case needed
            print("Loading model from file")
            self.model = tf.keras.models.load_model(model_path, compile=False)

            # Compile the model with basic settings
            print("Compiling model")
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}", exc_info=True)
            QMessageBox.warning(
                self,
                "Model Yükleme Hatası",
                f"Model yüklenirken bir hata oluştu. Hata: {str(e)}"
            )

    def predict_image(self, image_path):
        """Görüntü üzerinde tahmin yap"""
        print(f"Starting prediction for image: {image_path}")

        if self.model is None:
            print("Model not loaded, cannot make prediction")
            return "unknown", 0.0

        try:
            # Load and preprocess the image
            print("Loading and preprocessing image")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image from path: {image_path}")
                raise ValueError("Görüntü okunamadı")

            print(f"Original image shape: {img.shape}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            print(f"Resized image shape: {img.shape}")

            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            print(f"Preprocessed image shape: {img.shape}")

            # Make prediction with error handling
            print("Making prediction")
            prediction = self.model.predict(img, verbose=0)[0][0]
            print(f"Raw prediction value: {prediction}")

            # Update UI elements
            good_percent = prediction * 100
            bad_percent = (1 - prediction) * 100
            print(f"Good percentage: {good_percent:.2f}%, Bad percentage: {bad_percent:.2f}%")

            self.good_progress.setValue(int(good_percent))
            self.bad_progress.setValue(int(bad_percent))

            result = "good" if prediction > 0.5 else "bad"
            confidence = good_percent if prediction > 0.5 else bad_percent

            print(f"Final prediction: {result} with confidence: {confidence:.2f}%")

            self.result_label.setText(
                f"Tahmin: {'İYİ' if result == 'good' else 'KÖTÜ'}\n"
                f"Güven: %{confidence:.1f}"
            )

            # Update image preview
            print("Updating image preview")
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)
            self.current_image = image_path

            return result, confidence

        except Exception as e:
            print(f"Prediction error: {str(e)}", exc_info=True)
            QMessageBox.warning(
                self,
                "Tahmin Hatası",
                f"Görüntü analiz edilirken bir hata oluştu. Hata: {str(e)}"
            )
            return "unknown", 0.0

# Diğer sıkıntı foto çekildikten sonra bir daha çekilince predict