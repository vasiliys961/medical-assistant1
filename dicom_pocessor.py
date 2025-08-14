import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from datetime import datetime

class AdvancedDICOMProcessor:
    """Расширенный класс для обработки DICOM файлов"""
    
    def __init__(self):
        self.supported_modalities = [
            'CR', 'CT', 'MR', 'XA', 'RF', 'DX', 'MG', 'US', 'NM', 'PT'
        ]
    
    def read_dicom_file(self, file_path_or_buffer):
        """Чтение DICOM файла"""
        try:
            if hasattr(file_path_or_buffer, 'read'):
                # Если это file-like объект
                temp_path = "temp_dicom.dcm"
                with open(temp_path, "wb") as f:
                    f.write(file_path_or_buffer.getvalue())
                dicom_data = pydicom.dcmread(temp_path)
                os.remove(temp_path)
            else:
                # Если это путь к файлу
                dicom_data = pydicom.dcmread(file_path_or_buffer)
            
            return dicom_data
        except Exception as e:
            raise Exception(f"Ошибка при чтении DICOM файла: {e}")
    
    def extract_metadata(self, dicom_data):
        """Извлечение метаданных из DICOM файла"""
        metadata = {}
        
        # Основная информация о пациенте
        metadata['patient_name'] = str(getattr(dicom_data, 'PatientName', 'Не указано'))
        metadata['patient_id'] = str(getattr(dicom_data, 'PatientID', 'Не указано'))
        metadata['patient_birth_date'] = str(getattr(dicom_data, 'PatientBirthDate', 'Не указано'))
        metadata['patient_sex'] = str(getattr(dicom_data, 'PatientSex', 'Не указано'))
        metadata['patient_age'] = str(getattr(dicom_data, 'PatientAge', 'Не указано'))
        
        # Информация об исследовании
        metadata['study_date'] = str(getattr(dicom_data, 'StudyDate', 'Не указано'))
        metadata['study_time'] = str(getattr(dicom_data, 'StudyTime', 'Не указано'))
        metadata['study_description'] = str(getattr(dicom_data, 'StudyDescription', 'Не указано'))
        metadata['series_description'] = str(getattr(dicom_data, 'SeriesDescription', 'Не указано'))
        
        # Техническая информация
        metadata['modality'] = str(getattr(dicom_data, 'Modality', 'Не указано'))
        metadata['manufacturer'] = str(getattr(dicom_data, 'Manufacturer', 'Не указано'))
        metadata['manufacturer_model'] = str(getattr(dicom_data, 'ManufacturerModelName', 'Не указано'))
        metadata['body_part_examined'] = str(getattr(dicom_data, 'BodyPartExamined', 'Не указано'))
        metadata['patient_position'] = str(getattr(dicom_data, 'PatientPosition', 'Не указано'))
        
        # Параметры изображения
        if hasattr(dicom_data, 'pixel_array'):
            metadata['image_shape'] = dicom_data.pixel_array.shape
            metadata['bits_allocated'] = getattr(dicom_data, 'BitsAllocated', 'Не указано')
            metadata['bits_stored'] = getattr(dicom_data, 'BitsStored', 'Не указано')
            metadata['pixel_representation'] = getattr(dicom_data, 'PixelRepresentation', 'Не указано')
        
        # Дополнительные параметры для разных модальностей
        if metadata['modality'] == 'CT':
            metadata['slice_thickness'] = str(getattr(dicom_data, 'SliceThickness', 'Не указано'))
            metadata['kvp'] = str(getattr(dicom_data, 'KVP', 'Не указано'))
            metadata['exposure_time'] = str(getattr(dicom_data, 'ExposureTime', 'Не указано'))
        elif metadata['modality'] == 'MR':
            metadata['magnetic_field_strength'] = str(getattr(dicom_data, 'MagneticFieldStrength', 'Не указано'))
            metadata['repetition_time'] = str(getattr(dicom_data, 'RepetitionTime', 'Не указано'))
            metadata['echo_time'] = str(getattr(dicom_data, 'EchoTime', 'Не указано'))
        
        return metadata
    
    def get_pixel_array(self, dicom_data):
        """Получение массива пикселей из DICOM"""
        try:
            if not hasattr(dicom_data, 'pixel_array'):
                raise Exception("DICOM файл не содержит изображения")
            
            pixel_array = dicom_data.pixel_array
            
            # Применение параметров окна и уровня, если они есть
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = float(dicom_data.WindowCenter)
                window_width = float(dicom_data.WindowWidth)
                
                # Применение windowing
                pixel_array = self.apply_windowing(pixel_array, window_center, window_width)
            
            # Нормализация к диапазону 0-255
            pixel_array = self.normalize_pixel_array(pixel_array)
            
            return pixel_array
        except Exception as e:
            raise Exception(f"Ошибка при извлечении пикселей: {e}")
    
    def apply_windowing(self, pixel_array, window_center, window_width):
        """Применение параметров окна и уровня"""
        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2
        
        # Обрезка значений
        windowed_array = np.clip(pixel_array, min_value, max_value)
        
        return windowed_array
    
    def normalize_pixel_array(self, pixel_array):
        """Нормализация массива пикселей к диапазону 0-255"""
        # Преобразование к float для избежания overflow
        pixel_array = pixel_array.astype(np.float64)
        
        # Нормализация
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        
        if max_val > min_val:
            normalized = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(pixel_array, dtype=np.uint8)
        
        return normalized
    
    def enhance_image(self, pixel_array, enhancement_type='histogram_equalization'):
        """Улучшение качества изображения"""
        if enhancement_type == 'histogram_equalization':
            return cv2.equalizeHist(pixel_array)
        elif enhancement_type == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(pixel_array)
        elif enhancement_type == 'gaussian_blur':
            return cv2.GaussianBlur(pixel_array, (5, 5), 0)
        elif enhancement_type == 'median_filter':
            return cv2.medianBlur(pixel_array, 5)
        else:
            return pixel_array
    
    def analyze_image_quality(self, pixel_array):
        """Анализ качества изображения"""
        analysis = {}
        
        # Основная статистика
        analysis['mean_intensity'] = np.mean(pixel_array)
        analysis['std_intensity'] = np.std(pixel_array)
        analysis['min_intensity'] = np.min(pixel_array)
        analysis['max_intensity'] = np.max(pixel_array)
        
        # Контраст
        analysis['contrast'] = np.std(pixel_array)
        
        # Оценка шума (упрощенная)
        laplacian_var = cv2.Laplacian(pixel_array, cv2.CV_64F).var()
        analysis['noise_estimate'] = laplacian_var
        
        # Оценка резкости
        analysis['sharpness'] = laplacian_var
        
        # Энтропия (мера информационного содержания)
        hist, _ = np.histogram(pixel_array.flatten(), bins=256, range=[0, 256])
        hist = hist / hist.sum()  # Нормализация
        hist = hist[hist > 0]  # Удаление нулевых значений
        analysis['entropy'] = -np.sum(hist * np.log2(hist))
        
        # Градиент (мера краевого содержания)
        grad_x = cv2.Sobel(pixel_array, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(pixel_array, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        analysis['mean_gradient'] = np.mean(gradient_magnitude)
        
        return analysis
    
    def detect_anatomical_features(self, pixel_array, modality):
        """Простое обнаружение анатомических особенностей"""
        features = {}
        
        if modality in ['CR', 'DX']:  # Рентгенография
            # Обнаружение костных структур (высокая интенсивность)
            bone_threshold = np.percentile(pixel_array, 85)
            bone_mask = pixel_array > bone_threshold
            features['bone_area_percentage'] = np.sum(bone_mask) / pixel_array.size * 100
            
            # Обнаружение воздушных полостей (низкая интенсивность)
            air_threshold = np.percentile(pixel_array, 15)
            air_mask = pixel_array < air_threshold
            features['air_area_percentage'] = np.sum(air_mask) / pixel_array.size * 100
            
        elif modality == 'CT':
            # Анализ плотности тканей (в единицах Хаунсфилда, упрощенно)
            # Воздух: -1000, Жир: -100 до -50, Мягкие ткани: 20-70, Кость: >300
            features['tissue_distribution'] = {
                'air_like': np.sum(pixel_array < 50) / pixel_array.size * 100,
                'fat_like': np.sum((pixel_array >= 50) & (pixel_array < 100)) / pixel_array.size * 100,
                'soft_tissue_like': np.sum((pixel_array >= 100) & (pixel_array < 200)) / pixel_array.size * 100,
                'bone_like': np.sum(pixel_array >= 200) / pixel_array.size * 100
            }
        
        return features
    
    def generate_report(self, dicom_data, pixel_array, metadata, quality_analysis, anatomical_features):
        """Генерация отчета об анализе DICOM изображения"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'patient_info': {
                'name': metadata['patient_name'],
                'id': metadata['patient_id'],
                'age': metadata['patient_age'],
                'sex': metadata['patient_sex']
            },
            'study_info': {
                'date': metadata['study_date'],
                'modality': metadata['modality'],
                'body_part': metadata['body_part_examined'],
                'description': metadata['study_description']
            },
            'technical_parameters': {
                'image_shape': metadata['image_shape'],
                'manufacturer': metadata['manufacturer'],
                'model': metadata['manufacturer_model']
            },
            'quality_metrics': quality_analysis,
            'anatomical_analysis': anatomical_features,
            'recommendations': self.generate_recommendations(quality_analysis, anatomical_features, metadata)
        }
        
        return report
    
    def generate_recommendations(self, quality_analysis, anatomical_features, metadata):
        """Генерация рекомендаций на основе анализа"""
        recommendations = []
        
        # Рекомендации по качеству изображения
        if quality_analysis['contrast'] < 30:
            recommendations.append("Низкий контраст изображения. Рекомендуется настройка параметров визуализации.")
        
        if quality_analysis['noise_estimate'] > 1000:
            recommendations.append("Высокий уровень шума. Рекомендуется применение фильтров шумоподавления.")
        
        if quality_analysis['sharpness'] < 100:
            recommendations.append("Низкая резкость изображения. Проверьте параметры съемки.")
        
        # Рекомендации по модальности
        modality = metadata['modality']
        if modality in ['CR', 'DX'] and 'bone_area_percentage' in anatomical_features:
            if anatomical_features['bone_area_percentage'] < 10:
                recommendations.append("Низкая видимость костных структур. Проверьте экспозицию.")
        
        if not recommendations:
            recommendations.append("Качество изображения соответствует диагностическим требованиям.")
        
        return recommendations