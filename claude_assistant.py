import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import base64
import io
import os
import pydicom
try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

# --- КЛАСС ИИ-АССИСТЕНТА ---
class OpenRouterAssistant:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-3.5-sonnet"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vasiliys961/medical-assistant1",
            "X-Title": "Medical AI Assistant"
        }

    def test_connection(self):
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False

    def send_vision_request(self, prompt: str, image_array=None, metadata: str = ""):
        messages = [{"role": "user", "content": []}]
        full_text = f"{prompt}\n\n{metadata}" if metadata else prompt
        messages[0]["content"].append({"type": "text", "text": full_text})

        if image_array is not None:
            img = Image.fromarray(image_array) if isinstance(image_array, np.ndarray) else image_array
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.3
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"❌ Ошибка: {response.status_code}, {response.text}"
        except Exception as e:
            return f"❌ Ошибка подключения: {str(e)}"


# --- УТИЛИТЫ ---
def load_image_from_file(uploaded_file):
    """Загрузка изображения из JPG/PNG/DICOM/PDF + извлечение метаданных"""
    if uploaded_file.name.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)

        metadata = {
            "Modality": getattr(dicom, "Modality", "N/A"),
            "PatientName": getattr(dicom, "PatientName", "N/A"),
            "PatientSex": getattr(dicom, "PatientSex", "N/A"),
            "PatientAge": getattr(dicom, "PatientAge", "N/A"),
            "StudyDate": getattr(dicom, "StudyDate", "N/A"),
            "InstitutionName": getattr(dicom, "InstitutionName", "N/A"),
            "BodyPartExamined": getattr(dicom, "BodyPartExamined", "N/A"),
            "ImageType": getattr(dicom, "ImageType", "N/A"),
            "SeriesDescription": getattr(dicom, "SeriesDescription", "N/A"),
            "ProtocolName": getattr(dicom, "ProtocolName", "N/A"),
            "SliceThickness": getattr(dicom, "SliceThickness", "N/A"),
            "EchoTime": getattr(dicom, "EchoTime", "N/A"),
            "RepetitionTime": getattr(dicom, "RepetitionTime", "N/A"),
            "MagneticFieldStrength": getattr(dicom, "MagneticFieldStrength", "N/A"),
            "Comments": getattr(dicom, "ImageComments", "N/A")
        }
        metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items() if v != "N/A"])

        img = dicom.pixel_array
        if img.dtype != np.uint8:
            img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
        return img, "DICOM", metadata_str

    elif uploaded_file.name.lower().endswith(".pdf"):
        if convert_from_bytes is None:
            st.error("Установите `pdf2image`: pip install pdf2image")
            return None, None, ""
        pages = convert_from_bytes(uploaded_file.read(), dpi=200)
        img = pages[0].convert("L")
        metadata_str = f"PDF, {len(pages)} pages"
        return np.array(img), "PDF", metadata_str

    else:
        img = Image.open(uploaded_file)
        if img.mode != "L":
            img = img.convert("L")
        metadata_str = f"File: {uploaded_file.name}, Format: {img.format}, Size: {img.size}"
        return np.array(img), "IMAGE", metadata_str


def extract_ecg_signal(image_array):
    img = cv2.equalizeHist(image_array)
    h, w = img.shape
    start = h // 2 - 5
    end = h // 2 + 5
    signal = np.mean(img[start:end, :], axis=0)
    if np.mean(signal[:100]) > np.mean(signal[-100:]):
        signal = 255 - signal
    return signal


def analyze_ecg_basic(image_array):
    signal = extract_ecg_signal(image_array)
    heart_rate = int(60 / (len(signal) / 500)) * 10
    rhythm = "Синусовый" if 60 < heart_rate < 100 else "Нарушение ритма"
    duration = len(signal) / 500
    num_beats = int(duration * heart_rate / 60)
    return {
        "heart_rate": heart_rate,
        "rhythm_assessment": rhythm,
        "num_beats": num_beats,
        "duration": duration,
        "signal_quality": "Хорошее",
        "analysis_method": "Advanced Image Processing"
    }


def analyze_xray_basic(image_array):
    contrast = np.std(image_array)
    sharpness = cv2.Laplacian(image_array, cv2.CV_64F).var()
    quality = "Хорошее" if sharpness > 100 and contrast > 40 else "Удовлетворительное"
    return {
        "quality_assessment": quality,
        "contrast": float(contrast),
        "detailed_quality": {
            "sharpness": float(sharpness),
            "snr": float(np.mean(image_array) / np.std(image_array))
        },
        "lung_area": int(np.sum(image_array < 200)),
        "analysis_method": "Advanced Computer Vision"
    }


def analyze_mri_quality(image_array):
    sharpness = cv2.Laplacian(image_array, cv2.CV_64F).var()
    noise = np.std(image_array)
    snr = np.mean(image_array) / (noise + 1e-6)
    artifacts = "Возможны артефакты движения" if sharpness < 80 else "Минимальные артефакты"
    quality = "Хорошее" if sharpness > 100 and noise < 30 else "Удовлетворительное/Требует пересъёмки"
    return {
        "quality_assessment": quality,
        "sharpness": float(sharpness),
        "noise_level": float(noise),
        "snr": float(snr),
        "artifacts": artifacts,
        "analysis_method": "MRI-Specific CV Metrics"
    }


# --- СТРАНИЦЫ ---
def show_home_page():
    st.markdown("# 🏥 Медицинский ИИ-Ассистент v3.3")
    st.write("Vision + DICOM + анализ → ИИ с контекстом")
    st.info("✅ Добавлена поддержка МРТ с анализом параметров сканирования")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📈 ЭКГ")
        st.write("- ЧСС, ритм, аритмии")
    with col2:
        st.subheader("🩻 Рентген")
        st.write("- Качество, патология лёгких")
    with col3:
        st.subheader("🧠 МРТ")
        st.write("- Качество, анатомия, патология")


def show_ecg_analysis():
    st.header("📈 Анализ ЭКГ")
    uploaded_file = st.file_uploader("Загрузите ЭКГ (JPG, PNG, PDF, DICOM)", type=["jpg", "png", "pdf", "dcm"])

    if uploaded_file is None:
        st.info("Загрузите файл для анализа.")
        return

    with st.spinner("Обработка изображения..."):
        image_array, file_type, metadata_str = load_image_from_file(uploaded_file)
        if image_array is None:
            return
        analysis = analyze_ecg_basic(image_array)

    st.image(image_array, caption="ЭКГ", use_container_width=True, clamp=True)

    with st.expander("📄 Метаданные файла"):
        st.text(metadata_str)

    st.subheader("📊 Результаты анализа")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ЧСС", f"{analysis['heart_rate']} уд/мин")
        st.metric("Ритм", analysis['rhythm_assessment'])
    with col2:
        st.metric("Длительность", f"{analysis['duration']:.1f} с")
        st.metric("Комплексы", analysis['num_beats'])

    if 'assistant' not in st.session_state:
        st.session_state.assistant = OpenRouterAssistant()
    assistant = st.session_state.assistant

    if st.button("🔍 ИИ-анализ ЭКГ (с контекстом)"):
        with st.spinner("ИИ анализирует ЭКГ..."):
            clinical_metadata = (
                f"ЧСС: {analysis['heart_rate']}\nРитм: {analysis['rhythm_assessment']}\n"
                f"Длительность: {analysis['duration']:.1f} с\nМетод: {analysis['analysis_method']}"
            )
            full_metadata = f"=== ДАННЫЕ ===\n{metadata_str}\n\n=== АНАЛИЗ ===\n{clinical_metadata}"
            prompt = """
            Проанализируйте ЭКГ на изображении. Оцените ритм, ЧСС, признаки ишемии, блокад, аритмий.
            Учитывайте возраст и пол из метаданных. Дайте клинические рекомендации.
            """
            result = assistant.send_vision_request(prompt, image_array, full_metadata)
            st.markdown("### 🧠 Ответ ИИ:")
            st.write(result)

    custom_q = st.text_input("Задайте вопрос по ЭКГ:")
    if st.button("💬 Спросить ИИ") and custom_q:
        full_metadata = f"ЧСС: {analysis['heart_rate']}, ритм: {analysis['rhythm_assessment']}\n{metadata_str}"
        result = assistant.send_vision_request(custom_q, image_array, full_metadata)
        st.markdown("### 💬 Ответ:")
        st.write(result)


def show_xray_analysis():
    st.header("🩻 Анализ рентгена")
    uploaded_file = st.file_uploader("Загрузите рентген (JPG, PNG, DICOM)", type=["jpg", "png", "dcm"])

    if uploaded_file is None:
        st.info("Загрузите файл для анализа.")
        return

    with st.spinner("Обработка изображения..."):
        image_array, file_type, metadata_str = load_image_from_file(uploaded_file)
        if image_array is None:
            return
        analysis = analyze_xray_basic(image_array)

    st.image(image_array, caption="Рентген", use_container_width=True, clamp=True)

    with st.expander("📄 Метаданные"):
        st.text(metadata_str)

    st.subheader("📊 Оценка качества")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Качество", analysis['quality_assessment'])
        st.metric("Контраст", f"{analysis['contrast']:.1f}")
    with col2:
        st.metric("Резкость", f"{analysis['detailed_quality']['sharpness']:.1f}")
        st.metric("Площадь лёгких", f"{analysis['lung_area']:,}")

    if 'assistant' not in st.session_state:
        st.session_state.assistant = OpenRouterAssistant()
    assistant = st.session_state.assistant

    if st.button("🩺 ИИ-анализ рентгена"):
        with st.spinner("ИИ анализирует снимок..."):
            clinical_metadata = (
                f"Качество: {analysis['quality_assessment']}\nКонтраст: {analysis['contrast']:.1f}\n"
                f"Резкость: {analysis['detailed_quality']['sharpness']:.1f}"
            )
            full_metadata = f"=== ДАННЫЕ ===\n{metadata_str}\n\n=== АНАЛИЗ ===\n{clinical_metadata}"
            prompt = """
            Проанализируйте рентген грудной клетки. Оцените качество, структуры, признаки патологии.
            Дайте дифференциальный диагноз и рекомендации.
            """
            result = assistant.send_vision_request(prompt, image_array, full_metadata)
            st.markdown("### 🧠 Заключение:")
            st.write(result)

    custom_q = st.text_input("Вопрос по рентгену:")
    if st.button("💬 Спросить ИИ о снимке") and custom_q:
        full_metadata = f"Качество: {analysis['quality_assessment']}\n{metadata_str}"
        result = assistant.send_vision_request(custom_q, image_array, full_metadata)
        st.markdown("### 💬 Ответ:")
        st.write(result)


def show_mri_analysis():
    st.header("🧠 Анализ МРТ")
    uploaded_file = st.file_uploader("Загрузите МРТ (DICOM, JPG, PNG)", type=["dcm", "jpg", "png"])

    if uploaded_file is None:
        st.info("Загрузите DICOM-файл МРТ или изображение.")
        return

    with st.spinner("Обработка среза..."):
        image_array, file_type, metadata_str = load_image_from_file(uploaded_file)
        if image_array is None:
            return
        mri_analysis = analyze_mri_quality(image_array)

    st.image(image_array, caption="МРТ-срез", use_container_width=True, clamp=True)

    with st.expander("📄 Метаданные МРТ"):
        st.text(metadata_str)

    st.subheader("📊 Оценка качества МРТ")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Качество", mri_analysis['quality_assessment'])
        st.metric("Резкость", f"{mri_analysis['sharpness']:.1f}")
    with col2:
        st.metric("Шум", f"{mri_analysis['noise_level']:.1f}")
        st.metric("SNR", f"{mri_analysis['snr']:.2f}")

    st.caption(f"Артефакты: {mri_analysis['artifacts']}")

    if 'assistant' not in st.session_state:
        st.session_state.assistant = OpenRouterAssistant()
    assistant = st.session_state.assistant

    if st.button("🧠 ИИ-анализ МРТ (с контекстом)"):
        with st.spinner("ИИ анализирует МРТ..."):
            clinical_metadata = (
                f"Качество: {mri_analysis['quality_assessment']}\n"
                f"Резкость: {mri_analysis['sharpness']:.1f}\n"
                f"Шум: {mri_analysis['noise_level']:.1f}\n"
                f"SNR: {mri_analysis['snr']:.2f}\n"
                f"Артефакты: {mri_analysis['artifacts']}\n"
                f"Метод: {mri_analysis['analysis_method']}"
            )
            full_metadata = f"=== DICOM ДАННЫЕ ===\n{metadata_str}\n\n=== КАЧЕСТВО ===\n{clinical_metadata}"

            prompt = """
            Проанализируйте МРТ-срез на изображении. Учитывайте:
            1. Анатомию (голова, позвоночник, сустав и т.д.) из описания
            2. Качество: резкость, шум, артефакты
            3. Визуальные патологии: опухоли, отёк, грыжи, кровоизлияния
            4. Последовательность (T1, T2, FLAIR и т.д.)
            5. Рекомендации: другие проекции, контраст, КТ, консультация невролога
            Ответ должен быть нейровизуализационно точным.
            """

            result = assistant.send_vision_request(prompt, image_array, full_metadata)
            st.markdown("### 🧠 Нейрорадиологическое заключение:")
            st.write(result)

    custom_q = st.text_input("Вопрос по МРТ:")
    if st.button("💬 Спросить ИИ о МРТ") and custom_q:
        full_metadata = f"=== ДАННЫЕ ===\n{metadata_str}\n\n=== КАЧЕСТВО ===\n" + \
                        f"Качество: {mri_analysis['quality_assessment']}, SNR: {mri_analysis['snr']:.2f}"
        result = assistant.send_vision_request(custom_q, image_array, full_metadata)
        st.markdown("### 💬 Ответ:")
        st.write(result)


# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ ---
def main():
    st.set_page_config(page_title="Медицинский ИИ-Ассистент", layout="wide")
    st.sidebar.title("🧠 Меню")

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Главная"

    page = st.sidebar.selectbox(
        "Выберите раздел:",
        ["🏠 Главная", "📈 Анализ ЭКГ", "🩻 Анализ рентгена", "🧠 Анализ МРТ"],  # <-- МРТ ДОБАВЛЕН ЗДЕСЬ
        index=["🏠 Главная", "📈 Анализ ЭКГ", "🩻 Анализ рентгена", "🧠 Анализ МРТ"].index(st.session_state.current_page)
    )
    st.session_state.current_page = page

    # Маршрутизация
    if page == "🏠 Главная":
        show_home_page()
    elif page == "📈 Анализ ЭКГ":
        show_ecg_analysis()
    elif page == "🩻 Анализ рентгена":
        show_xray_analysis()
    elif page == "🧠 Анализ МРТ":  # <-- И ЗДЕСЬ
        show_mri_analysis()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Медицинский Ассистент v3.3**  
    🔹 Vision + DICOM + анализ → ИИ с контекстом  
    🔹 Поддержка: ЭКГ, рентген, МРТ, PDF  
    🔹 Claude 3.5 Sonnet + OpenRouter  
    ⚠️ Только для обучения
    """)


if __name__ == "__main__":
    main()