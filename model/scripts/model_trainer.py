import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt


class WallQualityModel:
    def __init__(self):
        # Tam yolları tanımla
        self.base_dir = r'C:\Users\gizem\Desktop\pythonProject\wall_analysis_project\model\data\training'
        self.save_dir = r'C:\Users\gizem\Desktop\pythonProject\wall_analysis_project\model\saved_models'
        self.batch_size = 32
        self.input_shape = (224, 224, 3)
        self.model = None

        # Save directory oluştur
        os.makedirs(self.save_dir, exist_ok=True)

    def create_data_generators(self):
        """Eğitim için data generator'ları oluştur"""
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2
        )

        print("Eğitim verileri yükleniyor...")
        train_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['augmented_bad', 'augmented_good'],
            subset='training'
        )

        print("\nValidation verileri yükleniyor...")
        validation_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['augmented_bad', 'augmented_good'],
            subset='validation'
        )

        return train_generator, validation_generator

    def build_model(self):
        """Model mimarisini oluştur"""
        print("\nModel oluşturuluyor...")
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Base model'i dondur
        base_model.trainable = False

        # Yeni katmanlar ekle
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Model derleme
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        print("Model başarıyla oluşturuldu!")

    def plot_training_history(self, history, title, filename):
        """Eğitim geçmişini görselleştir ve kaydet"""
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{title} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{title} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def train(self, epochs=20):
        """Modeli eğit"""
        if self.model is None:
            self.build_model()

        print("\nVeri yükleniyor...")
        train_generator, validation_generator = self.create_data_generators()

        print("\nEğitim başlıyor...")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.save_dir, 'wall_model_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        # Eğitim grafiklerini kaydet
        self.plot_training_history(history, "Initial Training", "training_history.png")

        return history

    def fine_tune(self, epochs=10):
        """Fine-tuning işlemi"""
        print("\nFine-tuning başlıyor...")

        # MobileNetV2 katmanını bul
        base_model = None
        for layer in self.model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break

        if base_model is not None:
            print("Base model bulundu, fine-tuning başlıyor...")
            base_model.trainable = True

            # İlk katmanları dondur, son katmanları eğitilebilir yap
            num_layers = len(base_model.layers)
            for i, layer in enumerate(base_model.layers):
                if i < num_layers - 30:  # Son 30 katman hariç dondur
                    layer.trainable = False
        else:
            print("Uyarı: Base model bulunamadı, fine-tuning atlanıyor...")
            return None

        # Modeli daha düşük learning rate ile yeniden derle
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Fine-tuning eğitimi
        train_generator, validation_generator = self.create_data_generators()

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
        )

        # Fine-tuning grafiklerini kaydet
        self.plot_training_history(history, "Fine Tuning", "fine_tuning_history.png")

        return history
def main():
    print("Model eğitimi başlatılıyor...")

    # Model nesnesini oluştur
    model = WallQualityModel()

    # İlk eğitimi yap
    print("\nİlk eğitim aşaması başlıyor...")
    history = model.train(epochs=20)

    # Fine-tuning yap
    print("\nFine-tuning aşaması başlıyor...")
    ft_history = model.fine_tune(epochs=10)

    # Son modeli kaydet
    model.model.save(os.path.join(model.save_dir, 'wall_model_final.h5'))
    print("\nEğitim tamamlandı! Model kaydedildi.")


if __name__ == "__main__":
    main()