import os
import json
import shutil
from datetime import datetime
import cv2
from pathlib import Path


class FeedbackManager:
    def __init__(self, base_dir='model/data'):
        """
        Feedback yönetim sistemini başlat

        Args:
            base_dir (str): Ana dizin yolu
        """
        self.base_dir = base_dir
        self.feedback_dir = os.path.join(base_dir, 'feedback')
        self.feedback_db = os.path.join(self.feedback_dir, 'feedback_history.json')

        # Feedback dizin yapısı
        self.correct_dir = os.path.join(self.feedback_dir, 'correct')
        self.incorrect_dir = os.path.join(self.feedback_dir, 'incorrect')

        # Alt klasörler
        self.correct_good_dir = os.path.join(self.correct_dir, 'good')
        self.correct_bad_dir = os.path.join(self.correct_dir, 'bad')
        self.incorrect_good_to_bad_dir = os.path.join(self.incorrect_dir, 'good_to_bad')
        self.incorrect_bad_to_good_dir = os.path.join(self.incorrect_dir, 'bad_to_good')

        # Dizinleri ve veritabanını oluştur
        self._create_directories()
        self._initialize_db()

    def _create_directories(self):
        """Gerekli dizin yapısını oluştur"""
        directories = [
            self.correct_good_dir,
            self.correct_bad_dir,
            self.incorrect_good_to_bad_dir,
            self.incorrect_bad_to_good_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _initialize_db(self):
        """Feedback veritabanını oluştur veya yükle"""
        if not os.path.exists(self.feedback_db):
            initial_data = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'last_retrain_date': None,
                'feedbacks': []
            }
            with open(self.feedback_db, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=4)

    def add_feedback(self, image_path, predicted_label, actual_label, confidence):
        """
        Yeni feedback ekle ve görüntüyü uygun klasöre kaydet

        Args:
            image_path (str): Görüntü dosyasının yolu
            predicted_label (str): Model tahmini ('good' veya 'bad')
            actual_label (str): Gerçek etiket ('good' veya 'bad')
            confidence (float): Model tahmin güveni (0-1 arası)

        Returns:
            tuple: (başarı durumu (bool), mesaj (str))
        """
        try:
            # Görüntünün var olduğunu kontrol et
            if not os.path.exists(image_path):
                return False, f"Görüntü bulunamadı: {image_path}"

            # Timestamp oluştur
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_filename = f'image_{timestamp}.jpg'

            # Doğru klasörü belirle
            if predicted_label == actual_label:
                if actual_label == 'good':
                    target_dir = self.correct_good_dir
                else:
                    target_dir = self.correct_bad_dir
            else:
                if predicted_label == 'good' and actual_label == 'bad':
                    target_dir = self.incorrect_good_to_bad_dir
                else:
                    target_dir = self.incorrect_bad_to_good_dir

            # Görüntüyü kopyala
            target_path = os.path.join(target_dir, new_filename)
            shutil.copy2(image_path, target_path)

            # Veritabanını güncelle
            self._update_db(predicted_label, actual_label, confidence, new_filename)

            return True, f"Feedback başarıyla kaydedildi: {target_path}"

        except Exception as e:
            return False, f"Feedback eklenirken hata oluştu: {str(e)}"

    def _update_db(self, predicted_label, actual_label, confidence, image_name):
        """Feedback veritabanını güncelle"""
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Yeni feedback girişi
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_name,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'confidence': float(confidence),
            'is_correct': predicted_label == actual_label
        }

        # İstatistikleri güncelle
        data['total_predictions'] += 1
        if predicted_label == actual_label:
            data['correct_predictions'] += 1
        else:
            data['incorrect_predictions'] += 1

        # Feedback'i ekle
        data['feedbacks'].append(feedback_entry)

        # Veritabanını kaydet
        with open(self.feedback_db, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def get_statistics(self):
        """
        Feedback istatistiklerini getir

        Returns:
            dict: İstatistik bilgileri
        """
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = data['total_predictions']
        accuracy = data['correct_predictions'] / total if total > 0 else 0

        return {
            'total_predictions': total,
            'correct_predictions': data['correct_predictions'],
            'incorrect_predictions': data['incorrect_predictions'],
            'accuracy': accuracy,
            'last_retrain_date': data['last_retrain_date']
        }

    def should_retrain(self, threshold=50):
        """
        Yeniden eğitim gerekip gerekmediğini kontrol et

        Args:
            threshold (int): Yeniden eğitim için gereken minimum yanlış tahmin sayısı

        Returns:
            bool: Yeniden eğitim gerekiyorsa True
        """
        stats = self.get_statistics()

        if stats['last_retrain_date'] is None:
            return stats['incorrect_predictions'] >= threshold

        # Son eğitimden sonraki yanlış tahminleri kontrol et
        last_retrain = datetime.fromisoformat(stats['last_retrain_date'])
        recent_incorrect = len([
            f for f in self.get_recent_feedbacks()
            if not f['is_correct'] and
               datetime.fromisoformat(f['timestamp']) > last_retrain
        ])

        return recent_incorrect >= threshold

    def get_recent_feedbacks(self, limit=100):
        """
        Son feedbackleri getir

        Args:
            limit (int): Getirilecek maksimum feedback sayısı

        Returns:
            list: Son feedbackler
        """
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return sorted(
            data['feedbacks'],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def mark_retrained(self):
        """Yeniden eğitim tarihini güncelle"""
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['last_retrain_date'] = datetime.now().isoformat()

        with open(self.feedback_db, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)


def main():
    """Test ve örnek kullanım"""
    feedback_manager = FeedbackManager()

    # Örnek feedback ekle
    success, message = feedback_manager.add_feedback(
        image_path='test_image.jpg',
        predicted_label='good',
        actual_label='bad',
        confidence=0.85
    )
    print(f"Feedback ekleme: {message}")

    # İstatistikleri göster
    stats = feedback_manager.get_statistics()
    print("\nFeedback İstatistikleri:")
    print(f"Toplam tahmin: {stats['total_predictions']}")
    print(f"Doğru tahmin: {stats['correct_predictions']}")
    print(f"Yanlış tahmin: {stats['incorrect_predictions']}")
    print(f"Doğruluk oranı: {stats['accuracy']:.2%}")

    # Yeniden eğitim kontrolü
    if feedback_manager.should_retrain():
        print("\nModel yeniden eğitilmeli!")


if __name__ == "__main__":
    main()