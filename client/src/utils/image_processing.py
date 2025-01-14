from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import numpy as np
import cv2
import os
import time
import requests


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False
        self.server_url = "http://192.168.1.118:5000"  # Raspberry Pi IP adresi

    def start(self):
        if not self.running:
            self.running = True
            super().start()
            return True
        return False

    def run(self):
        while self.running:
            try:
                # Raspberry Pi'den fotoğraf al
                response = requests.get(f"{self.server_url}/capture")
                if response.status_code == 200:
                    # Bytes verisini numpy array'e çevir
                    image_array = np.frombuffer(response.content, dtype=np.uint8)
                    # OpenCV ile decode et
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # BGR'den RGB'ye çevir
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape

                        # QImage oluştur
                        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                        self.frame_ready.emit(qt_image)

            except Exception as e:
                print(f"Error capturing frame: {str(e)}")

            # Her frame arasında kısa bir bekleme
            self.msleep(100)

    def stop(self):
        self.running = False
        self.wait()

    def capture_image(self, save_dir):
        try:
            # Raspberry Pi'den fotoğraf al
            response = requests.get(f"{self.server_url}/capture")
            if response.status_code != 200:
                return None, "Failed to capture image from camera"

            # Dosyayı kaydet
            timestamp = str(int(time.time()))
            filepath = os.path.join(save_dir, f"image_{timestamp}.jpg")

            with open(filepath, 'wb') as f:
                f.write(response.content)

            return filepath, "Image captured successfully"
        except Exception as e:
            return None, f"Error capturing image: {str(e)}"