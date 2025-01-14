import cv2
import time
import os


class Camera:
    def __init__(self):
        """Kamera nesnesini başlat"""
        try:
            self.camera = cv2.VideoCapture(0)

            # Kamera ayarlarını optimize et
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            # Kameranın başarıyla başlatılıp başlatılmadığını kontrol et
            if not self.camera.isOpened():
                raise Exception("Kamera başlatılamadı!")

            # Kameranın ısınması için birkaç kare al
            for _ in range(5):
                self.camera.read()
                time.sleep(0.1)

        except Exception as e:
            raise Exception(f"Kamera başlatılamadı: {str(e)}")

    def capture_frame(self):
        """Anlık görüntü al ve PIL Image olarak döndür"""
        try:
            # Birkaç kare al (buffer'ı temizlemek için)
            for _ in range(2):
                ret, frame = self.camera.read()

            if not ret:
                raise Exception("Görüntü alınamadı")

            return frame

        except Exception as e:
            raise Exception(f"Görüntü alınamadı: {str(e)}")

    def close(self):
        """Kamerayı kapat"""
        if hasattr(self, 'camera'):
            try:
                self.camera.release()
            except Exception as e:
                print(f"Kamera kapatılırken hata oluştu: {str(e)}")