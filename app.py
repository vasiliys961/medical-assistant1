import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json
import sqlite3
from datetime import datetime
import PyPDF2
import cv2
import base64
import io

# Попытка импорта модулей ИИ
try:
    from claude_assistant import OpenRouterAssistant
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Настройка страницы
st.set_page_config(
    page_title="Медицинский Ассистент с ИИ",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация базы данных
def init_database():
    conn = sqlite3.connect('medical_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            birth_date DATE,
            gender TEXT,
            phone TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_result TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Класс для обработки ЭКГ
class ECGProcessor:
    @staticmethod
    def process_csv_ecg(file):
        try:
            df = pd.read_csv(file)
            if 'time' in df.columns and 'voltage' in df.columns:
                return df
            elif len(df.columns) >= 2:
                df.columns = ['time', 'voltage'] + list(df.columns[2:])
                return df
            else:
                st.error("Неверный формат CSV файла для ЭКГ")
                return None
        except Exception as e:
            st.error(f"Ошибка при чтении CSV файла: {e}")
            return None
    
    @staticmethod
    def process_pdf_ecg(file):
        """Обработка PDF файла с ЭКГ данными"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Простое извлечение числовых данных
            import re
            numbers = re.findall(r'-?\d+\.?\d*', text)
            
            if len(numbers) >= 4:
                # Создаем искусственные данные на основе найденных чисел
                time_data = np.linspace(0, 10, len(numbers)//2)
                voltage_data = [float(x) for x in numbers[::2]][:len(time_data)]
                
                df = pd.DataFrame({
                    'time': time_data,
                    'voltage': voltage_data
                })
                return df
            else:
                st.warning("В PDF не найдены числовые данные ЭКГ")
                return None
                
        except Exception as e:
            st.error(f"Ошибка при чтении PDF файла: {e}")
            return None
    
    @staticmethod
    def process_image_ecg(file):
        """Обработка изображения ЭКГ (JPG, PNG)"""
        try:
            image = Image.open(file)
            image_array = np.array(image)
            
            # Конвертация в grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Простое извлечение профиля по горизонтали (имитация ЭКГ)
            height, width = gray.shape
            middle_row = height // 2
            
            # Берем значения из средней части изображения
            voltage_profile = gray[middle_row, :]
            
            # Нормализация и создание временной шкалы
            voltage_normalized = (voltage_profile - voltage_profile.mean()) / voltage_profile.std()
            time_data = np.linspace(0, 10, len(voltage_normalized))
            
            df = pd.DataFrame({
                'time': time_data,
                'voltage': voltage_normalized
            })
            
            return df, image_array
            
        except Exception as e:
            st.error(f"Ошибка при обработке изображения: {e}")
            return None, None
    
    @staticmethod
    def analyze_ecg(df):
        if df is None or df.empty:
            return None
        
        analysis = {}
        analysis['duration'] = df['time'].max() - df['time'].min()
        analysis['sample_rate'] = len(df) / analysis['duration'] if analysis['duration'] > 0 else 0
        analysis['mean_voltage'] = df['voltage'].mean()
        analysis['std_voltage'] = df['voltage'].std()
        analysis['min_voltage'] = df['voltage'].min()
        analysis['max_voltage'] = df['voltage'].max()
        
        # Простое определение ЧСС
        voltage_threshold = analysis['mean_voltage'] + 2 * analysis['std_voltage']
        peaks = df[df['voltage'] > voltage_threshold]
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks['time'].values)
            analysis['heart_rate'] = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            analysis['num_beats'] = len(peaks)
        else:
            analysis['heart_rate'] = 0
            analysis['num_beats'] = 0
        
        # Оценка ритма
        if analysis['heart_rate'] < 60:
            analysis['rhythm_assessment'] = "Брадикардия"
        elif analysis['heart_rate'] > 100:
            analysis['rhythm_assessment'] = "Тахикардия"
        else:
            analysis['rhythm_assessment'] = "Нормальный ритм"
        
        return analysis

# Класс для обработки рентгеновских снимков
class XRayProcessor:
    @staticmethod
    def process_image(file):
        """Обработка рентгеновского изображения"""
        try:
            image = Image.open(file)
            image_array = np.array(image)
            
            # Конвертация в grayscale если необходимо
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            metadata = {
                'format': image.format if hasattr(image, 'format') else 'Unknown',
                'mode': image.mode if hasattr(image, 'mode') else 'Unknown',
                'size': image.size if hasattr(image, 'size') else image_array.shape,
                'image_shape': image_array.shape
            }
            
            return image_array, metadata
        except Exception as e:
            st.error(f"Ошибка при обработке изображения: {e}")
            return None, None
    
    @staticmethod
    def analyze_xray(image_array, metadata):
        """Анализ рентгеновского снимка"""
        if image_array is None:
            return None
        
        analysis = {}
        analysis['image_statistics'] = {
            'mean_intensity': np.mean(image_array),
            'std_intensity': np.std(image_array),
            'min_intensity': np.min(image_array),
            'max_intensity': np.max(image_array),
            'shape': image_array.shape
        }
        
        # Анализ контраста
        analysis['contrast'] = np.std(image_array)
        
        # Анализ гистограммы
        hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])
        analysis['histogram'] = {
            'hist': hist.tolist(),
            'bins': bins.tolist()
        }
        
        # Простая оценка качества изображения
        if analysis['contrast'] < 30:
            analysis['quality_assessment'] = "Низкий контраст"
        elif analysis['contrast'] > 80:
            analysis['quality_assessment'] = "Высокий контраст"
        else:
            analysis['quality_assessment'] = "Нормальный контраст"
        
        analysis['metadata'] = metadata
        
        return analysis

# Главная страница
def show_home_page():
    st.header("Добро пожаловать в Медицинский Ассистент с ИИ!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 ИИ-Ассистент OpenRouter")
        st.write("**Возможности:**")
        st.write("- Интерпретация медицинских данных")
        st.write("- Объяснение результатов анализов")
        st.write("- Консультации по симптомам")
        st.write("- Образовательная поддержка")
        
        if AI_AVAILABLE:
            st.success("✅ ИИ-ассистент доступен")
        else:
            st.warning("⚠️ ИИ-ассистент недоступен")
    
    with col2:
        st.subheader("📊 Анализ данных")
        st.write("**Поддерживаемые форматы:**")
        st.write("📈 **ЭКГ:** CSV, PDF, JPG, PNG")
        st.write("🩻 **Рентген:** JPG, PNG, DICOM")
        st.write("🔬 **Анализы:** PDF, Excel, CSV")
        
        # Быстрые переходы
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("📈 Анализ ЭКГ", use_container_width=True):
                st.session_state.current_page = "📈 Анализ ЭКГ"
                st.rerun()
        with col2_2:
            if st.button("🩻 Рентген", use_container_width=True):
                st.session_state.current_page = "🩻 Анализ рентгена"
                st.rerun()

# Анализ ЭКГ
def show_ecg_analysis():
    st.header("📈 Анализ ЭКГ данных")
    
    uploaded_file = st.file_uploader(
        "Загрузите файл с данными ЭКГ",
        type=['csv', 'pdf', 'jpg', 'jpeg', 'png'],
        help="Поддерживаемые форматы: CSV, PDF, JPG, PNG"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Обработка файла..."):
            ecg_processor = ECGProcessor()
            df = None
            original_image = None
            
            if file_extension == 'csv':
                df = ecg_processor.process_csv_ecg(uploaded_file)
            elif file_extension == 'pdf':
                df = ecg_processor.process_pdf_ecg(uploaded_file)
            elif file_extension in ['jpg', 'jpeg', 'png']:
                result = ecg_processor.process_image_ecg(uploaded_file)
                if result and len(result) == 2:
                    df, original_image = result
                else:
                    df = None
            
            if df is not None:
                st.success(f"Файл {file_extension.upper()} успешно обработан!")
                
                # Анализ данных
                analysis = ecg_processor.analyze_ecg(df)
                
                # Сохранение в сессию для ИИ
                st.session_state.current_analysis = analysis
                
                # Отображение оригинального изображения если есть
                if original_image is not None:
                    st.subheader("Оригинальное изображение")
                    st.image(original_image, caption="Загруженное ЭКГ изображение", use_column_width=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("График ЭКГ")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['time'],
                        y=df['voltage'],
                        mode='lines',
                        name='ЭКГ сигнал',
                        line=dict(color='red', width=1)
                    ))
                    fig.update_layout(
                        title="Электрокардиограмма",
                        xaxis_title="Время (с)",
                        yaxis_title="Напряжение (мВ)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Результаты анализа")
                    if analysis:
                        st.metric("ЧСС", f"{analysis['heart_rate']:.0f} уд/мин")
                        st.metric("Ритм", analysis['rhythm_assessment'])
                        st.metric("Комплексы", analysis['num_beats'])
                        st.metric("Длительность", f"{analysis['duration']:.1f} с")
                        st.metric("Формат файла", file_extension.upper())
                
                # ИИ-анализ
                if AI_AVAILABLE and analysis:
                    st.markdown("---")
                    st.subheader("🤖 ИИ-Анализ ЭКГ")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        if st.button("Анализировать с помощью ИИ"):
                            try:
                                assistant = OpenRouterAssistant()
                                with st.spinner("ИИ анализирует ЭКГ..."):
                                    ai_response = assistant.analyze_ecg_data(analysis)
                                st.write("**Интерпретация ИИ:**")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"Ошибка ИИ-анализа: {e}")
                    
                    with col4:
                        custom_question = st.text_input("Задайте вопрос по ЭКГ:")
                        if st.button("Спросить ИИ") and custom_question:
                            try:
                                assistant = OpenRouterAssistant()
                                with st.spinner("ИИ отвечает..."):
                                    ai_response = assistant.analyze_ecg_data(analysis, custom_question)
                                st.write("**Ответ ИИ:**")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"Ошибка: {e}")

# Анализ рентгена
def show_xray_analysis():
    st.header("🩻 Анализ рентгеновских снимков")
    
    uploaded_file = st.file_uploader(
        "Загрузите рентгеновский снимок",
        type=['jpg', 'jpeg', 'png', 'dcm'],
        help="Поддерживаемые форматы: JPG, PNG, DICOM"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Обработка изображения..."):
            xray_processor = XRayProcessor()
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                image_array, metadata = xray_processor.process_image(uploaded_file)
            elif file_extension == 'dcm':
                st.warning("DICOM поддержка базовая - загрузите как JPG/PNG для лучшего анализа")
                try:
                    import pydicom
                    dicom_data = pydicom.dcmread(uploaded_file)
                    image_array = dicom_data.pixel_array
                    metadata = {'format': 'DICOM', 'shape': image_array.shape}
                except:
                    st.error("Ошибка чтения DICOM файла")
                    return
            else:
                st.error("Неподдерживаемый формат файла")
                return
            
            if image_array is not None:
                st.success("Изображение успешно обработано!")
                
                # Анализ изображения
                analysis = xray_processor.analyze_xray(image_array, metadata)
                
                # Сохранение для ИИ
                st.session_state.current_xray_analysis = analysis
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Рентгеновский снимок")
                    st.image(image_array, caption="Обработанное изображение", use_column_width=True, clamp=True)
                
                with col2:
                    st.subheader("Метаданные")
                    if metadata:
                        for key, value in metadata.items():
                            st.write(f"**{key}:** {value}")
                    
                    # Анализ изображения
                    if analysis:
                        st.subheader("Анализ изображения")
                        stats = analysis['image_statistics']
                        st.metric("Размер", f"{stats['shape']}")
                        st.metric("Средняя интенсивность", f"{stats['mean_intensity']:.1f}")
                        st.metric("Контраст", f"{analysis['contrast']:.1f}")
                        st.metric("Качество", analysis['quality_assessment'])
                
                # Гистограмма
                if analysis and 'histogram' in analysis:
                    st.subheader("Гистограмма интенсивности")
                    fig = px.histogram(
                        x=analysis['histogram']['bins'][:-1],
                        y=analysis['histogram']['hist'],
                        title="Распределение интенсивности пикселей"
                    )
                    fig.update_layout(
                        xaxis_title="Интенсивность",
                        yaxis_title="Количество пикселей"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # ИИ-анализ рентгена
                if AI_AVAILABLE and analysis:
                    st.markdown("---")
                    st.subheader("🤖 ИИ-Анализ рентгена")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        if st.button("Анализировать снимок с помощью ИИ"):
                            try:
                                assistant = OpenRouterAssistant()
                                with st.spinner("ИИ анализирует снимок..."):
                                    # Создаем контекст для рентгена
                                    context = f"""
Данные рентгеновского снимка:
- Качество изображения: {analysis['quality_assessment']}
- Контраст: {analysis['contrast']:.1f}
- Размер изображения: {analysis['image_statistics']['shape']}
- Средняя интенсивность: {analysis['image_statistics']['mean_intensity']:.1f}
"""
                                    ai_response = assistant.get_response(
                                        "Проанализируйте качество этого рентгеновского снимка и дайте рекомендации.",
                                        context
                                    )
                                st.write("**Анализ качества снимка:**")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"Ошибка ИИ-анализа: {e}")
                    
                    with col4:
                        custom_question = st.text_input("Вопрос по снимку:")
                        if st.button("Спросить ИИ о снимке") and custom_question:
                            try:
                                assistant = OpenRouterAssistant()
                                with st.spinner("ИИ отвечает..."):
                                    context = f"Рентгеновский снимок: качество {analysis['quality_assessment']}, контраст {analysis['contrast']:.1f}"
                                    ai_response = assistant.get_response(custom_question, context)
                                st.write("**Ответ ИИ:**")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"Ошибка: {e}")

# ИИ чат
def show_ai_chat():
    st.header("🤖 ИИ-Консультант")
    
    if not AI_AVAILABLE:
        st.error("ИИ-модуль недоступен. Проверьте файл claude_assistant.py")
        st.info("Убедитесь, что файл claude_assistant.py находится в той же папке")
        return
    
    try:
        assistant = OpenRouterAssistant()
        
        # Тест подключения
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 Тест подключения"):
                with st.spinner("Проверка подключения..."):
                    success, message = assistant.test_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            st.info("💡 Используется OpenRouter API с Claude 3 Sonnet")
        
        # Инициализация истории чата
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Быстрые вопросы
        st.subheader("⚡ Быстрые вопросы")
        quick_questions = [
            "Что такое ЭКГ?",
            "Объясни показатели анализа крови",
            "Что означает тахикардия?",
            "Нормы артериального давления"
        ]
        
        cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{i}"):
                    process_ai_message(question, assistant)
        
        st.markdown("---")
        
        # Отображение истории чата
        st.subheader("💬 Чат с ИИ-ассистентом")
        
        for message in st.session_state.chat_history[-10:]:  # Показываем последние 10 сообщений
            st.chat_message("user").write(message['user'])
            st.chat_message("assistant").write(message['assistant'])
        
        # Поле ввода сообщения
        user_input = st.chat_input("Задайте вопрос медицинскому ИИ-ассистенту...")
        
        if user_input:
            process_ai_message(user_input, assistant)
            
    except Exception as e:
        st.error(f"Ошибка инициализации ИИ: {e}")
        st.info("Проверьте наличие интернет-соединения и правильность API ключа")

def process_ai_message(user_message, assistant):
    """Обработка сообщения для ИИ"""
    # Добавляем сообщение пользователя
    st.chat_message("user").write(user_message)
    
    # Получаем ответ от ИИ
    with st.chat_message("assistant"):
        with st.spinner("ИИ думает..."):
            response = assistant.general_medical_consultation(user_message)
        st.write(response)
    
    # Сохраняем в историю
    st.session_state.chat_history.append({
        'user': user_message,
        'assistant': response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Ограничиваем историю
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]
    
    st.rerun()

# База данных пациентов
def show_patient_database():
    st.header("👤 База данных пациентов")
    
    tab1, tab2 = st.tabs(["Добавить пациента", "Поиск пациентов"])
    
    with tab1:
        st.subheader("Добавление нового пациента")
        
        with st.form("add_patient_form"):
            name = st.text_input("ФИО пациента*")
            birth_date = st.date_input("Дата рождения")
            gender = st.selectbox("Пол", ["Мужской", "Женский", "Не указан"])
            phone = st.text_input("Телефон")
            email = st.text_input("Email")
            
            submitted = st.form_submit_button("Добавить пациента")
            
            if submitted and name:
                conn = sqlite3.connect('medical_data.db')
                cursor = conn.cursor()
                
                try:
                    cursor.execute('''
                        INSERT INTO patients (name, birth_date, gender, phone, email)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (name, birth_date, gender, phone, email))
                    conn.commit()
                    st.success(f"Пациент {name} успешно добавлен!")
                except sqlite3.Error as e:
                    st.error(f"Ошибка при добавлении пациента: {e}")
                finally:
                    conn.close()
    
    with tab2:
        st.subheader("Поиск пациентов")
        
        search_term = st.text_input("Поиск по ФИО")
        
        conn = sqlite3.connect('medical_data.db')
        
        try:
            if search_term:
                query = "SELECT * FROM patients WHERE name LIKE ? ORDER BY created_at DESC"
                df_patients = pd.read_sql_query(query, conn, params=[f"%{search_term}%"])
            else:
                query = "SELECT * FROM patients ORDER BY created_at DESC LIMIT 10"
                df_patients = pd.read_sql_query(query, conn)
            
            if not df_patients.empty:
                st.dataframe(df_patients, use_container_width=True)
            else:
                st.info("Пациенты не найдены")
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
        finally:
            conn.close()

# Главная функция
def main():
    st.title("🏥 Медицинский Ассистент с ИИ")
    st.markdown("### Анализ медицинских данных с интегрированным ИИ-консультантом")
    
    # Инициализация базы данных
    init_database()
    
    # Боковая панель навигации
    st.sidebar.title("🧭 Навигация")
    
    # Обработка кнопок быстрого перехода
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Главная"
    
    page = st.sidebar.selectbox(
        "Выберите раздел:",
        [
            "🏠 Главная", 
            "📈 Анализ ЭКГ", 
            "🩻 Анализ рентгена",
            "🤖 ИИ-Консультант",
            "👤 База данных пациентов"
        ],
        index=[
            "🏠 Главная", 
            "📈 Анализ ЭКГ", 
            "🩻 Анализ рентгена",
            "🤖 ИИ-Консультант",
            "👤 База данных пациентов"
        ].index(st.session_state.current_page) if st.session_state.current_page in [
            "🏠 Главная", 
            "📈 Анализ ЭКГ", 
            "🩻 Анализ рентгена",
            "🤖 ИИ-Консультант",
            "👤 База данных пациентов"
        ] else 0
    )
    
    # Обновляем текущую страницу
    st.session_state.current_page = page
    
    # Маршрутизация страниц
    if page == "🏠 Главная":
        show_home_page()
    elif page == "📈 Анализ ЭКГ":
        show_ecg_analysis()
    elif page == "🩻 Анализ рентгена":
        show_xray_analysis()
    elif page == "🤖 ИИ-Консультант":
        show_ai_chat()
    elif page == "👤 База данных пациентов":
        show_patient_database()
    
    # Информация в боковой панели
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Медицинский Ассистент v2.0**
    
    🔹 Анализ ЭКГ (CSV, PDF, JPG, PNG)
    🔹 Анализ рентгена (JPG, PNG, DICOM)
    🔹 ИИ-консультации  
    🔹 База данных пациентов
    🔹 OpenRouter API
    
    ⚠️ Только для образовательных целей
    """)

if __name__ == "__main__":
    main()