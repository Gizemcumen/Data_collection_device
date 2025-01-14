import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QProgressBar, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import tensorflow as tf
import numpy as np
import cv2
import os


class PredictionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.current_image = None
        self.init_ui()  # Önce UI'ı oluştur
        self.load_model()  # Sonra modeli yükle

    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(r'C:\Users\gizem\Desktop\pythonProject\wall_analysis_project\model\saved_models\wall_model_final.h5')
            print(f"Model aranıyor: {model_path}")

            if os.path.exists(model_path):
                print(f"Model bulundu: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                print("Model başarıyla yüklendi!")
            else:
                print(f"Model dosyası bulunamadı: {model_path}")
                self.result_label.setText("Model yüklenemedi!")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            self.result_label.setText(f"Model yükleme hatası: {str(e)}")

    def init_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout()

        # Ana widget için stil
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # Sonuç başlığı
        self.result_title = QLabel("Analiz Sonucu")
        self.result_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.result_title.setAlignment(Qt.AlignCenter)
        self.result_title.setStyleSheet("color: #1a73e8; margin: 10px;")
        layout.addWidget(self.result_title)

        # Görüntü önizleme
        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #1a73e8;
                border-radius: 4px;
                padding: 2px;
                background-color: white;
            }
        """)
        self.image_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_preview)

        # İlerleme çubukları için container
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)

        # İyi durum göstergesi
        good_layout = QHBoxLayout()
        good_label = QLabel("İyi:")
        good_label.setFixedWidth(50)
        self.good_progress = QProgressBar()
        self.good_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                height: 20px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        good_layout.addWidget(good_label)
        good_layout.addWidget(self.good_progress)
        progress_layout.addLayout(good_layout)

        # Kötü durum göstergesi
        bad_layout = QHBoxLayout()
        bad_label = QLabel("Kötü:")
        bad_label.setFixedWidth(50)
        self.bad_progress = QProgressBar()
        self.bad_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #f44336;
                border-radius: 5px;
                text-align: center;
                height: 20px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #f44336;
            }
        """)
        bad_layout.addWidget(bad_label)
        bad_layout.addWidget(self.bad_progress)
        progress_layout.addLayout(bad_layout)

        layout.addWidget(progress_container)

        # Sonuç metni
        self.result_label = QLabel()
        self.result_label.setFont(QFont('Arial', 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                color: #202124;
                padding: 10px;
                background-color: white;
                border-radius: 4px;
                margin: 10px;
            }
        """)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict_image(self, image_path):
        """Görüntü üzerinde tahmin yap"""
        if self.model is None:
            self.result_label.setText("Model yüklü değil!")
            return None, 0

        try:
            # Görüntüyü yükle ve ön işle
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Görüntü yüklenemedi")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            # Tahmin yap
            prediction = self.model.predict(img)[0][0]

            # Progress bar'ları güncelle
            good_percent = prediction * 100
            bad_percent = (1 - prediction) * 100

            self.good_progress.setValue(int(good_percent))
            self.bad_progress.setValue(int(bad_percent))

            # Sonuç metnini güncelle
            result = "good" if prediction > 0.5 else "bad"
            confidence = good_percent if prediction > 0.5 else bad_percent

            self.result_label.setText(
                f"Tahmin: {'İYİ' if result == 'good' else 'KÖTÜ'}\n"
                f"Güven: %{confidence:.1f}"
            )

            # Görüntü önizlemeyi güncelle
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)

            return result, confidence

        except Exception as e:
            error_msg = f"Analiz hatası: {str(e)}"
            print(error_msg)
            self.result_label.setText(error_msg)
            return None, 0

    def reset_ui(self):
        """UI'ı sıfırla"""
        self.image_preview.clear()
        self.good_progress.setValue(0)
        self.bad_progress.setValue(0)
        self.result_label.clear()