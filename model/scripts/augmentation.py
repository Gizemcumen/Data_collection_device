import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import shutil


class DataAugmentor:
    def __init__(self):
        # Tam yol olarak ayarlayalım
        base_dir = r'C:\Users\gizem\Desktop\pythonProject\wall_analysis_project\model\data\training'
        self.base_dir = base_dir

        # Alt klasörler
        self.augmented_good_dir = os.path.join(base_dir, 'augmented_good')
        self.augmented_bad_dir = os.path.join(base_dir, 'augmented_bad')
        self.good_dir = os.path.join(base_dir, 'good')
        self.bad_dir = os.path.join(base_dir, 'bad')

        # Augmented klasörleri oluştur
        os.makedirs(self.augmented_good_dir, exist_ok=True)
        os.makedirs(self.augmented_bad_dir, exist_ok=True)

        # ImageDataGenerator ayarları
        self.image_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )

    def augment_images(self, source_dir, target_dir, num_augmented_per_image=5):
        """Her görüntü için belirtilen sayıda artırılmış versiyon oluştur"""
        print(f"\nİşleniyor: {source_dir}")

        # Görüntü dosyalarını listele
        image_files = [f for f in os.listdir(source_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Uyarı: {source_dir} klasöründe görüntü bulunamadı!")
            return

        # Orijinal görüntüleri kopyala
        print("Orijinal görüntüler kopyalanıyor...")
        for img_file in image_files:
            src_path = os.path.join(source_dir, img_file)
            dst_path = os.path.join(target_dir, img_file)
            shutil.copy2(src_path, dst_path)

        # Her görüntü için augmentation uygula
        print("Görüntüler artırılıyor...")
        for img_file in tqdm(image_files, desc="Görüntüler işleniyor"):
            try:
                # Görüntüyü yükle
                img_path = os.path.join(source_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Uyarı: {img_path} yüklenemedi")
                    continue

                # BGR'den RGB'ye çevir
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Her augmentation için
                for i in range(num_augmented_per_image):
                    # Görüntüyü augment et
                    img_array = np.expand_dims(img, 0)
                    augmented = next(self.image_generator.flow(
                        img_array,
                        batch_size=1
                    ))[0].astype(np.uint8)

                    # RGB'den BGR'ye çevir ve kaydet
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    output_path = os.path.join(
                        target_dir,
                        f'aug_{i}_{img_file}'
                    )
                    cv2.imwrite(output_path, augmented)

            except Exception as e:
                print(f"Hata oluştu ({img_file}): {str(e)}")

    def augment_dataset(self, augmentations_per_image=5):
        """Tüm veri setini işle"""
        # Good görüntülerini işle
        if os.path.exists(self.good_dir):
            print("\nİyi görüntüler işleniyor...")
            self.augment_images(self.good_dir, self.augmented_good_dir, augmentations_per_image)
        else:
            print(f"Uyarı: {self.good_dir} klasörü bulunamadı!")

        # Bad görüntülerini işle
        if os.path.exists(self.bad_dir):
            print("\nKötü görüntüler işleniyor...")
            self.augment_images(self.bad_dir, self.augmented_bad_dir, augmentations_per_image)
        else:
            print(f"Uyarı: {self.bad_dir} klasörü bulunamadı!")

        # İstatistikleri göster
        print("\nVeri Seti İstatistikleri:")
        print(f"Orijinal iyi görüntüler: {len(os.listdir(self.good_dir))}")
        print(f"Artırılmış iyi görüntüler: {len(os.listdir(self.augmented_good_dir))}")
        print(f"Orijinal kötü görüntüler: {len(os.listdir(self.bad_dir))}")
        print(f"Artırılmış kötü görüntüler: {len(os.listdir(self.augmented_bad_dir))}")


def main():
    # Veri artırma işlemini başlat
    augmentor = DataAugmentor()
    # Her görüntü için 10 artırılmış versiyon oluştur
    augmentor.augment_dataset(augmentations_per_image=10)


if __name__ == "__main__":
    main()