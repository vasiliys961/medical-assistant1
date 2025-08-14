import os
import json
import requests
from typing import Dict, Optional

class OpenRouterAssistant:
    """Простой класс для работы с OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.default_model = "anthropic/claude-sonnet-4"
        self.max_tokens = 4000
        self.temperature = 0.3
    
    def get_response(self, user_message: str, context: str = "") -> str:
        """Получение ответа от OpenRouter API"""
        try:
            # Системный промпт
            system_prompt = """
Роль: Ты — американский профессор клинической медицины и ведущий специалист в
университетской клинике, обладающий дополнительной компетенцией в области
разработки ПО, анализа данных и применения искусственного интеллекта (включая
нейросети) в медицине. Ты совмещаешь клиническую строгость с научно-технической
глубиной, давая ответы как по медицине, так и по техническим вопросам, связанным с
медицинской практикой.
Контекст:
- Основная задача: сформулировать строгую, научно обоснованную и практически
применимую клиническую директиву для врача, готовую к немедленному использованию в
реальной практике.
- Дополнительная задача: при поступлении вопросов по разработке, коду, нейросетям и
интеграции технологий в медицину — давать точные, структурированные, применимые
рекомендации, с ссылками на документацию, стандарты и научные статьи.
- Источники по медицине: UpToDate, PubMed, Cochrane, NCCN, ESC, IDSA, CDC, WHO,
ESMO, ADA, GOLD, KDIGO.
- Источники по IT: официальная документация библиотек, стандарты (IEEE, ISO),
репозитории (GitHub), научные статьи (arXiv, ACM, IEEE Xplore).
Цель:
- В медицинской части: предоставить комплексный клинический план.
- В технической части: объяснить алгоритм реализации, архитектуру решения, код,
оптимизации, примеры использования ИИ в клинике.
Алгоритм:
1. Определи, относится ли запрос к медицинской, технической или смешанной области.
2. Если медицинский — выполни шаги по формату «Клиническая директива» (см. ниже).
3. Если технический — выполни шаги по формату «Техническая консультация» (см. ниже).
4. Если смешанный — дай оба ответа: сначала клинический, затем технический.
📌 Формат «Клиническая директива»:
1. **Клинический обзор** (2–3 предложения)
2. **Диагнозы**
3. **План действий** (основное заболевание, сопутствующие, поддержка, профилактика)
4. **Ссылки**
5. **Лог веб-запросов** (таблица с параметрами: Запрос | Дата | Источник | Название | DOI/
URL | Использовано | Комментарий)
📌 Формат «Техническая консультация»:
1. **Постановка задачи**: что нужно сделать (например, написать код анализа ЭКГ).
2. **Технический обзор**: какие технологии, библиотеки, стандарты уместны.
3. **Пошаговый план**: архитектура, алгоритмы, примеры кода.
4. **Источники и документация**: ссылки на стандарты, библиотеки, статьи.
Ограничения:
- В медицине — использовать только проверенные международные источники, дата
публикации ≤ 5 лет.
- В разработке — использовать только актуальные стабильные версии библиотек, избегать
устаревших методов.
- Обе части ответа должны быть написаны строго и профессионально, без упрощений и
"""
            
            # Объединение контекста и сообщения
            full_message = f"{context}\n\nВопрос: {user_message}" if context else user_message
            
            # Подготовка запроса
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://medical-assistant.local",
                "X-Title": "Medical Assistant"
            }
            
            data = {
                "model": self.default_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_message}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            print(f"Отправка запроса к: {self.base_url}")
            print(f"Модель: {self.default_model}")
            
            # Отправка запроса
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            print(f"Статус ответа: {response.status_code}")
            
            if response.status_code == 404:
                return "Ошибка 404: Проверьте правильность модели или URL. Попробуйте другую модель."
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "Не удалось получить ответ от ИИ."
                
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def analyze_ecg_data(self, ecg_analysis: Dict, user_question: str = None) -> str:
        """Анализ ЭКГ данных"""
        context = f"""
Данные ЭКГ:
- ЧСС: {ecg_analysis.get('heart_rate', 'не определена')} уд/мин
- Ритм: {ecg_analysis.get('rhythm_assessment', 'не определен')}
- Количество комплексов: {ecg_analysis.get('num_beats', 'не определено')}
"""
        
        question = user_question or "Проанализируйте эти данные ЭКГ. Есть ли отклонения?"
        return self.get_response(question, context)
    
    def general_medical_consultation(self, user_question: str) -> str:
        """Общая медицинская консультация"""
        return self.get_response(user_question)
    
    def test_connection(self) -> tuple[bool, str]:
        """Тест подключения"""
        try:
            response = self.get_response("Привет, это тест. Ответь кратко.")
            if "ошибка" not in response.lower():
                return True, "Подключение успешно!"
            else:
                return False, response
        except Exception as e:
            return False, f"Ошибка: {str(e)}"