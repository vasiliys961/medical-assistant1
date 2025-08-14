# medical_formats_processor.py - Добавьте в папку проекта

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import streamlit as st
import io
import base64
from scipy import signal
import matplotlib.pyplot as plt

# Попытка импорта DICOM
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    st.warning("⚠️ pydicom не установлен. Для работы с DICOM: pip install pydicom")

# Попытка импорта для работы с медицинскими сигналами
try:
    import wfdb  # PhysioNet WFDB format
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

class MedicalFormatProcessor:
    """Профессиональный процессор медицинских форматов"""
    
    def __init__(self):
        self.supported_formats = {
            'ECG': ['csv', 'txt', 'dat', 'edf', 'hea', 'mat'],
            'Images': ['dcm', 'dicom', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            'Documents': ['pdf', 'doc', 'docx']
        }
    
    def detect_format_quality(self, file, file_extension):
        """Определение качества и типа медицинского файла"""
        quality_report = {
            'format_type': 'unknown',
            'professional_grade': False,
            'quality_score': 0,
            'recommendations': [],
            'warnings': []
        }
        
        try:
            if file_extension.lower() in ['dcm', 'dicom']:
                quality_report.update(self._assess_dicom_quality(file))
            elif file_extension.lower() in ['jpg', 'jpeg', 'png', 'tiff']:
                quality_report.update(self._assess_image_quality(file))
            elif file_extension.lower() in ['csv', 'txt', 'dat']:
                quality_report.update(self._assess_signal_quality(file))
            elif file_extension.lower() == 'edf':
                quality_report.update(self._assess_edf_quality(file))
            
            # Добавляем общие рекомендации
            self._add_general_recommendations(quality_report, file_extension)
            
        except Exception as e:
            quality_report['warnings'].append(f"Ошибка анализа формата: {str(e)}")
        
        return quality_report
    
    def _assess_dicom_quality(self, file):
        """Оценка качества DICOM файла"""
        if not DICOM_AVAILABLE:
            return {
                'format_type': 'dicom',
                'professional_grade': True,
                'quality_score': 95,
                'warnings': ['pydicom не установлен - используется базовая обработка']
            }
        
        try:
            dicom_data = pydicom.dcmread(file)
            
            quality_assessment = {
                'format_type': 'dicom',
                'professional_grade': True,
                'quality_score': 85
            }
            
            # Проверяем ключевые DICOM теги
            if hasattr(dicom_data, 'Modality'):
                quality_assessment['modality'] = dicom_data.Modality
                quality_assessment['quality_score'] += 5
            
            if hasattr(dicom_data, 'InstitutionName'):
                quality_assessment['institution'] = dicom_data.InstitutionName
                quality_assessment['quality_score'] += 3
            
            if hasattr(dicom_data, 'StudyDate'):
                quality_assessment['study_date'] = dicom_data.StudyDate
                quality_assessment['quality_score'] += 2
            
            # Проверяем качество пиксельных данных
            if hasattr(dicom_data, 'pixel_array'):
                pixel_array = dicom_data.pixel_array
                quality_assessment['image_shape'] = pixel_array.shape
                quality_assessment['bit_depth'] = dicom_data.get('BitsStored', 'Unknown')
                
                if len(pixel_array.shape) >= 2:
                    if min(pixel_array.shape[:2]) >= 512:
                        quality_assessment['quality_score'] = min(100, quality_assessment['quality_score'] + 5)
                    elif min(pixel_array.shape[:2]) < 256:
                        quality_assessment['warnings'] = ['Низкое разрешение изображения']
                        quality_assessment['quality_score'] -= 10
            
            quality_assessment['recommendations'] = [
                "DICOM - профессиональный медицинский формат",
                "Содержит метаданные для точной диагностики",
                "Рекомендуется для клинического использования"
            ]
            
            return quality_assessment
            
        except Exception as e:
            return {
                'format_type': 'dicom',
                'professional_grade': True,
                'quality_score': 60,
                'warnings': [f"Ошибка чтения DICOM: {str(e)}"]
            }
    
    def _assess_image_quality(self, file):
        """Оценка качества обычных изображений"""
        try:
            image = Image.open(file)
            
            quality_assessment = {
                'format_type': 'consumer_image',
                'professional_grade': False,
                'quality_score': 30  # Базовая оценка для потребительских форматов
            }
            
            # Анализ разрешения
            width, height = image.size
            total_pixels = width * height
            
            if total_pixels >= 2000000:  # 2MP+
                quality_assessment['quality_score'] += 20
            elif total_pixels >= 1000000:  # 1MP+
                quality_assessment['quality_score'] += 10
            else:
                quality_assessment['warnings'] = ['Очень низкое разрешение']
                quality_assessment['quality_score'] -= 10
            
            # Анализ формата
            if image.format == 'PNG':
                quality_assessment['quality_score'] += 10
                quality_assessment['recommendations'] = ['PNG - хороший выбор для медицинских изображений']
            elif image.format == 'JPEG':
                quality_assessment['quality_score'] += 5
                quality_assessment['warnings'] = quality_assessment.get('warnings', []) + \
                    ['JPEG сжатие может влиять на точность анализа']
            elif image.format == 'TIFF':
                quality_assessment['quality_score'] += 15
                quality_assessment['recommendations'] = ['TIFF - отличный формат для медицины']
            
            # Анализ битности
            if image.mode in ['L', 'RGB']:
                if image.mode == 'L':  # Grayscale
                    quality_assessment['quality_score'] += 5
            elif image.mode in ['I', 'F']:  # 32-bit
                quality_assessment['quality_score'] += 10
                quality_assessment['professional_grade'] = True
            
            quality_assessment['image_info'] = {
                'size': f"{width}x{height}",
                'format': image.format,
                'mode': image.mode,
                'total_pixels': total_pixels
            }
            
            return quality_assessment
            
        except Exception as e:
            return {
                'format_type': 'image',
                'professional_grade': False,
                'quality_score': 0,
                'warnings': [f"Ошибка анализа изображения: {str(e)}"]
            }
    
    def _assess_signal_quality(self, file):
        """Оценка качества файлов с сигналами"""
        try:
            # Пытаемся прочитать как CSV
            df = pd.read_csv(file)
            
            quality_assessment = {
                'format_type': 'signal_data',
                'professional_grade': False,
                'quality_score': 40
            }
            
            # Анализ структуры данных
            if len(df.columns) >= 2:
                quality_assessment['quality_score'] += 15
                
                # Проверяем наличие временных меток
                time_columns = [col for col in df.columns if 'time' in col.lower() or 't' == col.lower()]
                if time_columns:
                    quality_assessment['quality_score'] += 10
                    quality_assessment['has_timestamps'] = True
                
                # Проверяем частоту дискретизации
                if len(df) > 1000:  # Достаточно данных для анализа
                    quality_assessment['quality_score'] += 10
                    quality_assessment['sample_count'] = len(df)
                    
                    if len(df) > 5000:  # Высокое качество записи
                        quality_assessment['quality_score'] += 10
                        quality_assessment['professional_grade'] = True
                
                # Проверяем именование колонок
                medical_keywords = ['ecg', 'voltage', 'lead', 'signal', 'mv', 'amplitude']
                medical_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in medical_keywords)]
                
                if medical_columns:
                    quality_assessment['quality_score'] += 15
                    quality_assessment['medical_columns'] = medical_columns
                    quality_assessment['recommendations'] = [
                        'Найдены медицинские термины в названиях колонок',
                        'Данные выглядят как профессиональная запись'
                    ]
                else:
                    quality_assessment['warnings'] = [
                        'Названия колонок не содержат медицинских терминов',
                        'Убедитесь в правильности интерпретации данных'
                    ]
            
            quality_assessment['data_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
            
            return quality_assessment
            
        except Exception as e:
            return {
                'format_type': 'signal_data',
                'professional_grade': False,
                'quality_score': 0,
                'warnings': [f"Ошибка чтения данных: {str(e)}"]
            }
    
    def _assess_edf_quality(self, file):
        """Оценка качества EDF файлов (European Data Format)"""
        quality_assessment = {
            'format_type': 'edf_medical',
            'professional_grade': True,
            'quality_score': 90,
            'recommendations': [
                'EDF - стандартный медицинский формат',
                'Используется в профессиональном медицинском оборудовании',
                'Содержит метаданные о пациенте и записи'
            ]
        }
        
        # Здесь можно добавить специфичный анализ EDF
        # Требует библиотеки pyedflib или mne
        
        return quality_assessment
    
    def _add_general_recommendations(self, quality_report, file_extension):
        """Добавление общих рекомендаций по формату"""
        recommendations = quality_report.get('recommendations', [])
        warnings = quality_report.get('warnings', [])
        
        # Рекомендации по форматам
        format_recommendations = {
            'dcm': ['Золотой стандарт медицинской визуализации', 'Сохраняет все метаданные'],
            'png': ['Без потерь качества', 'Хорошо для ЭКГ изображений'],
            'tiff': ['Профессиональный формат', 'Поддерживает высокую битность'],
            'jpg': ['Сжатие с потерями', 'Может искажать медицинские данные'],
            'csv': ['Простой формат данных', 'Убедитесь в правильности кодировки'],
            'edf': ['Медицинский стандарт', 'Используется в ЭЭГ/ЭКГ оборудовании']
        }
        
        if file_extension.lower() in format_recommendations:
            recommendations.extend(format_recommendations[file_extension.lower()])
        
        # Предупреждения для непрофессиональных форматов
        consumer_formats = ['jpg', 'jpeg', 'bmp', 'gif']
        if file_extension.lower() in consumer_formats:
            warnings.append('Потребительский формат может не подходить для точной диагностики')
        
        quality_report['recommendations'] = recommendations
        quality_report['warnings'] = warnings

class SmartFormatConverter:
    """Интеллектуальный конвертер медицинских форматов"""
    
    @staticmethod
    def enhance_consumer_image(image_array):
        """Улучшение потребительских изображений для медицинского анализа"""
        enhanced = image_array.copy()
        
        # Увеличение контраста для медицинского анализа
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        # CLAHE для улучшения локального контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
        
        # Подавление шума
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    @staticmethod
    def convert_dicom_to_standard(dicom_file):
        """Конвертация DICOM в стандартный формат для анализа"""
        if not DICOM_AVAILABLE:
            return None
        
        try:
            dicom_data = pydicom.dcmread(dicom_file)
            
            # Получение пиксельных данных
            pixel_array = dicom_data.pixel_array
            
            # Применение LUT преобразований если есть
            if hasattr(dicom_data, 'RescaleSlope'):
                pixel_array = apply_modality_lut(pixel_array, dicom_data)
            
            if hasattr(dicom_data, 'WindowCenter'):
                pixel_array = apply_voi_lut(pixel_array, dicom_data)
            
            # Нормализация для отображения
            if pixel_array.dtype != np.uint8:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            return pixel_array, dicom_data
            
        except Exception as e:
            st.error(f"Ошибка конвертации DICOM: {e}")
            return None

def show_format_quality_assessment(file, file_extension):
    """Отображение оценки качества формата"""
    processor = MedicalFormatProcessor()
    
    with st.spinner("Анализ качества формата..."):
        quality_report = processor.detect_format_quality(file, file_extension)
    
    # Отображение результатов
    st.subheader("📋 Оценка качества формата")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = quality_report['quality_score']
        if score >= 80:
            st.success(f"**Оценка качества:** {score}/100")
        elif score >= 60:
            st.warning(f"**Оценка качества:** {score}/100")
        else:
            st.error(f"**Оценка качества:** {score}/100")
    
    with col2:
        if quality_report['professional_grade']:
            st.success("✅ **Профессиональный формат**")
        else:
            st.warning("⚠️ **Потребительский формат**")
    
    with col3:
        format_type = quality_report['format_type'].replace('_', ' ').title()
        st.info(f"**Тип:** {format_type}")
    
    # Рекомендации
    if quality_report.get('recommendations'):
        st.markdown("**💡 Рекомендации:**")
        for rec in quality_report['recommendations']:
            st.write(f"• {rec}")
    
    # Предупреждения
    if quality_report.get('warnings'):
        st.markdown("**⚠️ Предупреждения:**")
        for warning in quality_report['warnings']:
            st.warning(f"• {warning}")
    
    # Детальная информация
    with st.expander("🔍 Подробная информация"):
        for key, value in quality_report.items():
            if key not in ['recommendations', 'warnings', 'quality_score', 'professional_grade']:
                st.write(f"**{key}:** {value}")
    
    return quality_report

def show_format_upgrade_suggestions(quality_report):
    """Предложения по улучшению формата"""
    st.subheader("🚀 Рекомендации по улучшению")
    
    score = quality_report['quality_score']
    
    if score < 60:
        st.error("**Критически низкое качество формата!**")
        st.markdown("""
        ### 🆘 Срочные рекомендации:
        1. **Для ЭКГ:** Используйте оборудование с прямым экспортом в CSV/EDF
        2. **Для рентгена:** Получите DICOM файлы от радиологического оборудования
        3. **Для изображений:** Сканируйте с разрешением 300+ DPI в PNG формате
        4. **Избегайте:** Фото на телефон, JPEG с высоким сжатием
        """)
    
    elif score < 80:
        st.warning("**Формат требует улучшения**")
        st.markdown("""
        ### 💡 Рекомендации по улучшению:
        1. **Увеличьте разрешение** изображений до 1000+ пикселей
        2. **Используйте PNG вместо JPEG** для медицинских изображений
        3. **Добавьте метаданные** в CSV файлы (время, частота дискретизации)
        4. **Проверьте качество** исходного сигнала/изображения
        """)
    
    else:
        st.success("**Отличное качество формата!**")
        st.markdown("""
        ### ✅ Ваш формат соответствует медицинским стандартам:
        - Подходит для профессионального анализа
        - Содержит достаточно данных для точной диагностики
        - Рекомендуется для клинического использования
        """)
    
    # Специфичные рекомендации по типу
    format_type = quality_report.get('format_type', '')
    
    if 'consumer' in format_type:
        st.info("""
        **🔄 Переход на профессиональные форматы:**
        - **DICOM** для медицинских изображений
        - **EDF** для электрофизиологических сигналов
        - **HL7 FHIR** для медицинских данных
        """)
    
    elif format_type == 'dicom':
        st.success("""
        **🏆 Вы используете золотой стандарт медицинской визуализации!**
        - Сохраняет все метаданные
        - Стандарт DICOM обеспечивает совместимость
        - Подходит для архивирования в PACS системах
        """)

# Функция для интеграции в основное приложение
def integrate_format_assessment(uploaded_file, file_extension):
    """Интеграция оценки формата в основное приложение"""
    
    # Показываем оценку качества формата
    quality_report = show_format_quality_assessment(uploaded_file, file_extension)
    
    # Предупреждение для низкокачественных форматов
    if quality_report['quality_score'] < 50:
        st.error("""
        ⚠️ **ВНИМАНИЕ: Низкое качество формата!**
        
        Результаты анализа могут быть неточными. 
        Рекомендуется использовать профессиональные медицинские форматы.
        """)
        
        # Предлагаем улучшения
        show_format_upgrade_suggestions(quality_report)
        
        # Спрашиваем, продолжать ли анализ
        if not st.button("⚠️ Продолжить анализ (на свой страх и риск)"):
            st.stop()
    
    elif quality_report['quality_score'] < 70:
        st.warning("""
        ⚠️ **Формат требует улучшения**
        
        Для более точного анализа рекомендуется улучшить качество данных.
        """)
        show_format_upgrade_suggestions(quality_report)
    
    else:
        st.success("✅ **Формат соответствует медицинским стандартам**")
    
    return quality_report