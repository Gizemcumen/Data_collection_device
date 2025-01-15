from PyQt5.QtCore import QThread, pyqtSignal
import tensorflow as tf

class ModelLoader(QThread):
    model_loaded = pyqtSignal(object)  # Signal to emit when model is loaded
    model_error = pyqtSignal(str)  # Signal to emit if there's an error

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            # Configure GPU memory
            tf.keras.backend.clear_session()
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            if gpu_devices:
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)

            # Load and compile model
            model = tf.keras.models.load_model(self.model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            self.model_loaded.emit(model)

        except Exception as e:
            self.model_error.emit(str(e))