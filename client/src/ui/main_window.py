import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QLabel,
                             QMessageBox, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QGroupBox, QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor
from utils.image_processing import CameraThread
from ui.prediction_panel import PredictionPanel
import shutil


class LabelDialog(QDialog):
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.result = None
        self.image_path = image_path
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Label Image')
        self.setFixedSize(800, 700)
        self.setStyleSheet("""
           QDialog {
               background-color: #f0f2f5;
           }
           QPushButton {
               background-color: #1a73e8;
               color: white;
               border: none;
               padding: 8px 15px;
               border-radius: 4px;
               font-size: 14px;
               min-width: 120px;
           }
           QPushButton:hover {
               background-color: #1557b0;
           }
           QLabel {
               color: #202124;
               font-size: 14px;
           }
       """)

        layout = QVBoxLayout()

        # Display captured image
        if self.image_path:
            image_label = QLabel()
            pixmap = QPixmap(self.image_path)
            scaled_pixmap = pixmap.scaled(700, 500, Qt.KeepAspectRatio)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)

        # Question label
        question = QLabel('How would you rate this wall?')
        question.setFont(QFont('Arial', 12))
        question.setAlignment(Qt.AlignCenter)
        layout.addWidget(question)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        good_btn = QPushButton('Good')
        good_btn.clicked.connect(lambda: self.set_result('good'))

        bad_btn = QPushButton('Bad')
        bad_btn.clicked.connect(lambda: self.set_result('bad'))

        btn_layout.addStretch()
        btn_layout.addWidget(good_btn)
        btn_layout.addWidget(bad_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def set_result(self, result):
        self.result = result
        self.accept()


import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from ui.prediction_panel import PredictionPanel


class ResultDialog(QDialog):
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.image_path = image_path
        self.feedback = None
        self.prediction = None
        self.confidence = 0

        # UI'ı başlat
        self.init_ui()

        # Prediction panel'i başlat
        self.prediction_panel = PredictionPanel(self)
        self.layout().insertWidget(1, self.prediction_panel)  # Image label'dan sonra ekle

        # Tahmin yap
        if self.image_path:
            try:
                self.prediction, self.confidence = self.prediction_panel.predict_image(self.image_path)
            except Exception as e:
                print(f"Tahmin hatası: {str(e)}")
                self.prediction = "unknown"
                self.confidence = 0

    def init_ui(self):
        """UI bileşenlerini oluştur"""
        self.setWindowTitle('Analiz Sonucu')
        self.setFixedSize(800, 800)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f2f5;
            }
            QPushButton {
                background-color: #1a73e8;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
            QPushButton.correct {
                background-color: #0F9D58;
            }
            QPushButton.incorrect {
                background-color: #DB4437;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Görüntü önizleme
        if self.image_path and os.path.exists(self.image_path):
            try:
                image_label = QLabel()
                pixmap = QPixmap(self.image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(700, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled_pixmap)
                    image_label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(image_label)
                else:
                    print("Pixmap yüklenemedi")
            except Exception as e:
                print(f"Görüntü yükleme hatası: {str(e)}")

        # Feedback bölümü
        feedback_container = QWidget()
        feedback_layout = QVBoxLayout(feedback_container)

        feedback_label = QLabel("Bu tahmin doğru muydu?")
        feedback_label.setFont(QFont('Arial', 12))
        feedback_label.setAlignment(Qt.AlignCenter)
        feedback_layout.addWidget(feedback_label)

        button_layout = QHBoxLayout()

        correct_btn = QPushButton('Doğru')
        correct_btn.setProperty('class', 'correct')
        correct_btn.clicked.connect(lambda: self.set_feedback(True))

        incorrect_btn = QPushButton('Yanlış')
        incorrect_btn.setProperty('class', 'incorrect')
        incorrect_btn.clicked.connect(lambda: self.set_feedback(False))

        button_layout.addStretch()
        button_layout.addWidget(correct_btn)
        button_layout.addWidget(incorrect_btn)
        button_layout.addStretch()

        feedback_layout.addLayout(button_layout)
        layout.addWidget(feedback_container)

        self.setLayout(layout)

    def set_feedback(self, is_correct):
        """Feedback'i kaydet ve dialog'u kapat"""
        try:
            self.feedback = is_correct
            print(f"Feedback kaydedildi: {is_correct}")
            self.accept()
        except Exception as e:
            print(f"Feedback kaydetme hatası: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize camera
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)

        # Setup directories
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        project_root = os.path.dirname(current_dir)
        self.model_data_dir = os.path.join(project_root, 'model', 'data')
        self.training_dir = os.path.join(self.model_data_dir, 'training')
        self.test_dir = os.path.join(self.model_data_dir, 'test')
        self.feedback_dir = os.path.join(self.model_data_dir, 'feedback')

        # Create directories
        for path in [
            os.path.join(self.training_dir, 'good'),
            os.path.join(self.training_dir, 'bad'),
            os.path.join(self.test_dir, 'good'),
            os.path.join(self.test_dir, 'bad'),

            # Yeni feedback klasörleri
            os.path.join(self.feedback_dir, 'correct', 'good'),
            os.path.join(self.feedback_dir, 'correct', 'bad'),
            os.path.join(self.feedback_dir, 'incorrect', 'good_to_bad'),
            os.path.join(self.feedback_dir, 'incorrect', 'bad_to_good')
        ]:
            os.makedirs(path, exist_ok=True)

        self.current_mode = 'training'
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Wall Quality Analysis System')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("""
           QMainWindow {
               background-color: #f0f2f5;
           }
           QPushButton {
               background-color: #1a73e8;
               color: white;
               border: none;
               padding: 8px 15px;
               border-radius: 4px;
               font-size: 14px;
               min-width: 120px;
           }
           QPushButton:hover {
               background-color: #1557b0;
           }
           QPushButton:disabled {
               background-color: #a0a0a0;
           }
           QLabel {
               color: #202124;
           }
           QGroupBox {
               border: 2px solid #e0e0e0;
               border-radius: 6px;
               margin-top: 1em;
               padding: 15px;
               background-color: white;
           }
           QRadioButton {
               color: #202124;
               spacing: 8px;
               padding: 5px;
           }
       """)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Welcome message
        welcome_label = QLabel('Wall Quality Analysis System')
        welcome_label.setFont(QFont('Arial', 24, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
           QLabel {
               color: #1a73e8;
               margin: 20px;
           }
       """)
        layout.addWidget(welcome_label)

        # Mode selection
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(30)

        self.training_radio = QRadioButton("Training Mode")
        self.test_radio = QRadioButton("Test Mode")
        self.training_radio.setFont(QFont('Arial', 12))
        self.test_radio.setFont(QFont('Arial', 12))
        self.training_radio.setChecked(True)

        mode_layout.addWidget(self.training_radio)
        mode_layout.addWidget(self.test_radio)
        mode_group.setLayout(mode_layout)

        # Group buttons
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.training_radio)
        self.mode_group.addButton(self.test_radio)
        self.mode_group.buttonClicked.connect(self.mode_changed)

        # Camera preview
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
           QLabel {
               background-color: #2c2c2c;
               border: 2px solid #1a73e8;
               border-radius: 8px;
               padding: 2px;
           }
       """)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.start_camera_btn = QPushButton('Start Camera')
        self.start_camera_btn.clicked.connect(self.toggle_camera)

        self.capture_btn = QPushButton('Capture')
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)

        btn_layout.addStretch()
        btn_layout.addWidget(self.start_camera_btn)
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addStretch()

        # Status label
        self.status_label = QLabel('Camera: Off')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
           QLabel {
               color: #1a73e8;
               font-size: 14px;
               font-weight: bold;
               padding: 10px;
               background-color: white;
               border-radius: 4px;
           }
       """)

        # Add everything to main layout
        layout.addWidget(mode_group)
        layout.addWidget(self.preview_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.status_label)

        main_widget.setLayout(layout)

    def toggle_camera(self):
        if self.start_camera_btn.text() == 'Start Camera':
            if self.camera_thread.start():
                self.start_camera_btn.setText('Stop Camera')
                self.capture_btn.setEnabled(True)
                self.status_label.setText('Camera: On')
        else:
            self.camera_thread.stop()
            self.start_camera_btn.setText('Start Camera')
            self.capture_btn.setEnabled(False)
            self.status_label.setText('Camera: Off')
            self.preview_label.clear()

    def update_frame(self, qimage):
        scaled_pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio)
        self.preview_label.setPixmap(scaled_pixmap)

    def mode_changed(self, button):
        self.current_mode = 'training' if button == self.training_radio else 'test'



    def capture_image(self):
        save_dir = self.training_dir if self.current_mode == 'training' else self.test_dir
        filepath, message = self.camera_thread.capture_image(save_dir)

        if not filepath:
            QMessageBox.critical(self, "Error", message)
            return

        try:
            if self.current_mode == 'training':
                # Training mode
                dialog = LabelDialog(self, filepath)
                if dialog.exec_() == QDialog.Accepted and dialog.result:
                    new_dir = os.path.join(save_dir, dialog.result)
                    new_path = os.path.join(new_dir, os.path.basename(filepath))
                    os.rename(filepath, new_path)
                    QMessageBox.information(self, "Success", "Image saved and labeled successfully!")
            else:
                # Test mode
                dialog = ResultDialog(parent=self, image_path=filepath)
                if dialog.exec_() == QDialog.Accepted:
                    if dialog.feedback is not None:
                        # Handle feedback and save image
                        if dialog.prediction:
                            if dialog.feedback:  # Prediction was correct
                                feedback_dir = os.path.join(self.feedback_dir, 'correct', dialog.prediction)
                            else:  # Prediction was incorrect
                                if dialog.prediction == 'good':
                                    feedback_dir = os.path.join(self.feedback_dir, 'incorrect', 'good_to_bad')
                                else:
                                    feedback_dir = os.path.join(self.feedback_dir, 'incorrect', 'bad_to_good')

                            # Save to feedback directory
                            feedback_path = os.path.join(feedback_dir, os.path.basename(filepath))
                            shutil.copy2(filepath, feedback_path)

                            # Save to test directory
                            test_dir = os.path.join(self.test_dir,
                                                    dialog.prediction if dialog.feedback else
                                                    ('bad' if dialog.prediction == 'good' else 'good'))
                            test_path = os.path.join(test_dir, os.path.basename(filepath))
                            os.rename(filepath, test_path)

                            QMessageBox.information(self, "Success",
                                                    "Image saved and feedback recorded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    def closeEvent(self, event):
        self.camera_thread.stop()
        super().closeEvent(event)