# enhanced_processors.py
import cv2
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, filtfilt
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans

class AdvancedECGProcessor:
    """Продвинутый процессор для анализа ЭКГ из изображений"""
    
    def __init__(self):
        self.sampling_rate = 500  # Частота дискретизации
        
    def preprocess_image(self, image):
        """Предобработка изображения ЭКГ"""
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Увеличение контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Удаление шума с помощью bilateral filter
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Морфологические операции для удаления артефактов
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_grid_and_remove(self, image):
        """Обнаружение и удаление сетки ЭКГ"""
        # Обнаружение горизонтальных и вертикальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Извлечение горизонтальных линий
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Извлечение вертикальных линий
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Комбинация сетки
        grid = cv2.add(horizontal_lines, vertical_lines)
        
        # Удаление сетки из изображения
        no_grid = cv2.subtract(image, grid)
        
        return no_grid, grid
    
    def extract_ecg_signal(self, image):
        """Извлечение ЭКГ сигнала из изображения"""
        # Предобработка
        processed = self.preprocess_image(image)
        
        # Удаление сетки
        no_grid, grid = self.detect_grid_and_remove(processed)
        
        # Бинаризация для выделения ЭКГ линии
        _, binary = cv2.threshold(no_grid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary  # Инвертирование
        
        # Скелетизация для получения тонкой линии
        skeleton = morphology.skeletonize(binary > 0)
        
        # Извлечение координат ЭКГ линии
        y_coords, x_coords = np.where(skeleton)
        
        if len(x_coords) == 0:
            return None, None
        
        # Сортировка по x-координате и группировка
        points = list(zip(x_coords, y_coords))
        points.sort()
        
        # Создание сигнала путем усреднения y-координат для каждого x
        signal_dict = {}
        for x, y in points:
            if x not in signal_dict:
                signal_dict[x] = []
            signal_dict[x].append(y)
        
        # Создание финального сигнала
        x_signal = []
        y_signal = []
        
        for x in sorted(signal_dict.keys()):
            x_signal.append(x)
            y_signal.append(np.mean(signal_dict[x]))
        
        # Нормализация и инвертирование (ЭКГ обычно инвертирована в изображении)
        y_signal = np.array(y_signal)
        y_signal = image.shape[0] - y_signal  # Инвертирование
        y_signal = (y_signal - np.mean(y_signal)) / np.std(y_signal)  # Нормализация
        
        # Создание временной шкалы
        duration = 10.0  # Предполагаем 10 секунд
        time_signal = np.linspace(0, duration, len(x_signal))
        
        return time_signal, y_signal
    
    def filter_signal(self, signal, lowcut=0.5, highcut=50):
        """Фильтрация ЭКГ сигнала"""
        if signal is None or len(signal) < 10:
            return signal
            
        nyquist = self.sampling_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal
        except:
            return signal
    
    def detect_qrs_complexes(self, time, signal):
        """Обнаружение QRS комплексов"""
        if signal is None or len(signal) < 10:
            return [], []
        
        # Фильтрация сигнала
        filtered = self.filter_signal(signal)
        
        # Поиск пиков
        # Вычисляем динамический порог
        threshold = np.std(filtered) * 2 + np.mean(filtered)
        
        # Минимальное расстояние между пиками (0.3 секунды)
        min_distance = int(0.3 * len(signal) / (time[-1] - time[0]))
        
        peaks, properties = find_peaks(
            filtered, 
            height=threshold,
            distance=min_distance,
            prominence=np.std(filtered) * 0.5
        )
        
        peak_times = [time[peak] for peak in peaks if peak < len(time)]
        peak_values = [signal[peak] for peak in peaks if peak < len(signal)]
        
        return peak_times, peak_values
    
    def calculate_heart_rate(self, peak_times):
        """Расчет частоты сердечных сокращений"""
        if len(peak_times) < 2:
            return 0, 0, "Недостаточно данных"
        
        # RR интервалы
        rr_intervals = np.diff(peak_times)
        
        # Средняя ЧСС
        mean_hr = 60 / np.mean(rr_intervals)
        
        # Вариабельность ЧСС
        hrv = np.std(rr_intervals) * 1000  # в миллисекундах
        
        # Оценка ритма
        if mean_hr < 60:
            rhythm = "Брадикардия"
        elif mean_hr > 100:
            rhythm = "Тахикардия"
        elif np.std(rr_intervals) > 0.15:
            rhythm = "Аритмия"
        else:
            rhythm = "Синусовый ритм"
        
        return mean_hr, hrv, rhythm
    
    def analyze_ecg_advanced(self, image):
        """Комплексный анализ ЭКГ"""
        try:
            # Извлечение сигнала
            time, signal = self.extract_ecg_signal(image)
            
            if time is None or signal is None:
                return None
            
            # Обнаружение QRS комплексов
            peak_times, peak_values = self.detect_qrs_complexes(time, signal)
            
            # Расчет ЧСС
            hr, hrv, rhythm = self.calculate_heart_rate(peak_times)
            
            # Анализ качества сигнала
            signal_quality = self.assess_signal_quality(signal)
            
            analysis = {
                'time': time.tolist(),
                'signal': signal.tolist(),
                'peak_times': peak_times,
                'peak_values': peak_values,
                'heart_rate': hr,
                'hrv': hrv,
                'rhythm': rhythm,
                'num_beats': len(peak_times),
                'duration': time[-1] - time[0],
                'signal_quality': signal_quality,
                'analysis_method': 'Advanced Image Processing'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Ошибка анализа ЭКГ: {e}")
            return None
    
    def assess_signal_quality(self, signal):
        """Оценка качества сигнала"""
        if signal is None or len(signal) < 10:
            return "Плохое"
        
        # Отношение сигнал/шум
        signal_power = np.var(signal)
        noise_estimate = np.var(np.diff(signal))
        snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 0
        
        if snr > 20:
            return "Отличное"
        elif snr > 15:
            return "Хорошее"
        elif snr > 10:
            return "Удовлетворительное"
        else:
            return "Плохое"


class AdvancedXRayProcessor:
    """Продвинутый процессор для анализа рентгеновских снимков"""
    
    def __init__(self):
        self.lung_cascade = None
        self.rib_cascade = None
        
    def preprocess_xray(self, image):
        """Предобработка рентгеновского снимка"""
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Нормализация интенсивности
        normalized = cv2.equalizeHist(gray)
        
        # CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(normalized)
        
        # Гауссово размытие для уменьшения шума
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return denoised
    
    def detect_anatomy_regions(self, image):
        """Обнаружение анатомических областей"""
        processed = self.preprocess_xray(image)
        
        # Сегментация с помощью watershed
        markers = np.zeros_like(processed, dtype=np.int32)
        
        # Порог для выделения костей (яркие области)
        bone_threshold = filters.threshold_otsu(processed) * 1.2
        bone_mask = processed > bone_threshold
        
        # Порог для выделения легких (темные области)
        lung_threshold = filters.threshold_otsu(processed) * 0.7
        lung_mask = processed < lung_threshold
        
        # Морфологические операции
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=100)
        lung_mask = morphology.remove_small_objects(lung_mask, min_size=500)
        
        # Маркировка регионов
        markers[bone_mask] = 2
        markers[lung_mask] = 1
        
        # Watershed сегментация
        labels = segmentation.watershed(-processed, markers, mask=processed > 0)
        
        return labels, bone_mask, lung_mask
    
    def analyze_lung_fields(self, image, lung_mask):
        """Анализ легочных полей"""
        if lung_mask is None:
            return {}
        
        # Разделение на левое и правое легкое
        labeled_lungs = measure.label(lung_mask)
        regions = measure.regionprops(labeled_lungs)
        
        lung_analysis = {
            'num_regions': len(regions),
            'total_lung_area': np.sum(lung_mask),
            'lung_regions': []
        }
        
        for i, region in enumerate(regions):
            if region.area > 1000:  # Фильтрация мелких областей
                # Вычисление характеристик
                centroid = region.centroid
                area = region.area
                perimeter = region.perimeter
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Анализ текстуры в области легкого
                minr, minc, maxr, maxc = region.bbox
                lung_roi = image[minr:maxr, minc:maxc]
                
                texture_analysis = self.analyze_texture(lung_roi)
                
                lung_analysis['lung_regions'].append({
                    'region_id': i,
                    'centroid': centroid,
                    'area': area,
                    'circularity': circularity,
                    'texture': texture_analysis
                })
        
        return lung_analysis
    
    def analyze_texture(self, roi):
        """Анализ текстуры области"""
        if roi.size == 0:
            return {}
        
        # Статистические характеристики
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        # Локальная бинарная структура (упрощенная версия)
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        texture_features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'edge_density': np.mean(edge_magnitude),
            'homogeneity': 1.0 / (1.0 + std_intensity) if std_intensity > 0 else 1.0
        }
        
        return texture_features
    
    def detect_abnormalities(self, image, lung_mask):
        """Обнаружение возможных патологий"""
        abnormalities = []
        
        if lung_mask is None:
            return abnormalities
        
        # Поиск областей с аномальной яркостью
        lung_roi = image[lung_mask]
        mean_lung_intensity = np.mean(lung_roi)
        std_lung_intensity = np.std(lung_roi)
        
        # Поиск слишком ярких областей (возможные инфильтраты)
        bright_threshold = mean_lung_intensity + 2 * std_lung_intensity
        bright_areas = (image > bright_threshold) & lung_mask
        
        # Поиск слишком темных областей (возможные полости)
        dark_threshold = mean_lung_intensity - 2 * std_lung_intensity
        dark_areas = (image < dark_threshold) & lung_mask
        
        # Анализ найденных областей
        bright_regions = measure.regionprops(measure.label(bright_areas))
        dark_regions = measure.regionprops(measure.label(dark_areas))
        
        for region in bright_regions:
            if region.area > 50:  # Минимальный размер
                abnormalities.append({
                    'type': 'bright_area',
                    'location': region.centroid,
                    'area': region.area,
                    'severity': 'moderate' if region.area < 200 else 'high'
                })
        
        for region in dark_regions:
            if region.area > 50:
                abnormalities.append({
                    'type': 'dark_area',
                    'location': region.centroid,
                    'area': region.area,
                    'severity': 'moderate' if region.area < 200 else 'high'
                })
        
        return abnormalities
    
    def analyze_xray_advanced(self, image):
        """Комплексный анализ рентгеновского снимка"""
        try:
            # Предобработка
            processed = self.preprocess_xray(image)
            
            # Сегментация анатомических областей
            labels, bone_mask, lung_mask = self.detect_anatomy_regions(processed)
            
            # Анализ легочных полей
            lung_analysis = self.analyze_lung_fields(processed, lung_mask)
            
            # Поиск аномалий
            abnormalities = self.detect_abnormalities(processed, lung_mask)
            
            # Общая оценка качества снимка
            quality_assessment = self.assess_image_quality(image, processed)
            
            analysis = {
                'image_quality': quality_assessment,
                'lung_analysis': lung_analysis,
                'abnormalities': abnormalities,
                'bone_area': np.sum(bone_mask) if bone_mask is not None else 0,
                'lung_area': np.sum(lung_mask) if lung_mask is not None else 0,
                'total_pixels': image.size,
                'analysis_method': 'Advanced Computer Vision'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Ошибка анализа рентгена: {e}")
            return None
    
    def assess_image_quality(self, original, processed):
        """Оценка качества изображения"""
        # Анализ контраста
        contrast = np.std(processed)
        
        # Анализ резкости (Laplacian variance)
        laplacian_var = cv2.Laplacian(processed, cv2.CV_64F).var()
        
        # Анализ шума
        noise_estimate = np.std(cv2.GaussianBlur(processed, (3,3), 0) - processed)
        
        # SNR оценка
        signal_power = np.var(processed)
        snr = 10 * np.log10(signal_power / (noise_estimate**2)) if noise_estimate > 0 else float('inf')
        
        # Комплексная оценка
        if contrast > 50 and laplacian_var > 100 and snr > 20:
            quality = "Отличное"
        elif contrast > 30 and laplacian_var > 50 and snr > 15:
            quality = "Хорошее"
        elif contrast > 20 and laplacian_var > 25 and snr > 10:
            quality = "Удовлетворительное"
        else:
            quality = "Требует улучшения"
        
        return {
            'overall_quality': quality,
            'contrast': contrast,
            'sharpness': laplacian_var,
            'noise_level': noise_estimate,
            'snr': snr
        }


class MedicalImageEnhancer:
    """Класс для улучшения качества медицинских изображений"""
    
    @staticmethod
    def enhance_contrast(image, alpha=1.5, beta=0):
        """Улучшение контраста"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def denoise_image(image, h=10, templateWindowSize=7, searchWindowSize=21):
        """Подавление шума"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)
        else:
            return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    
    @staticmethod
    def sharpen_image(image):
        """Повышение резкости"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def adaptive_histogram_equalization(image):
        """Адаптивная гистограммная эквализация"""
        if len(image.shape) == 3:
            # Для цветных изображений - конвертируем в LAB и применяем к L каналу
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)