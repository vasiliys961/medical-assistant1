import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import PyPDF2
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class ReferenceRange:
    """Класс для хранения референсных диапазонов"""
    min_value: float
    max_value: float
    unit: str
    gender: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    condition: Optional[str] = None

@dataclass
class LabResult:
    """Класс для хранения результата анализа"""
    parameter: str
    value: float
    unit: str
    reference_range: Optional[ReferenceRange] = None
    status: str = "Не определено"
    comment: Optional[str] = None
    method: Optional[str] = None

class AdvancedLabProcessor:
    """Расширенный класс для обработки лабораторных данных"""
    
    def __init__(self):
        self.reference_ranges = self._load_reference_ranges()
        self.parameter_synonyms = self._load_parameter_synonyms()
        
    def _load_reference_ranges(self) -> Dict[str, List[ReferenceRange]]:
        """Загрузка референсных диапазонов"""
        ranges = {
            # Общий анализ крови
            'hemoglobin': [
                ReferenceRange(120, 160, 'г/л', gender='М'),
                ReferenceRange(115, 145, 'г/л', gender='Ж'),
                ReferenceRange(110, 140, 'г/л', age_min=60)
            ],
            'erythrocytes': [
                ReferenceRange(4.0, 5.1, '10^12/л', gender='М'),
                ReferenceRange(3.7, 4.7, '10^12/л', gender='Ж')
            ],
            'leukocytes': [
                ReferenceRange(4.0, 9.0, '10^9/л')
            ],
            'platelets': [
                ReferenceRange(150, 400, '10^9/л')
            ],
            'hematocrit': [
                ReferenceRange(39, 49, '%', gender='М'),
                ReferenceRange(35, 45, '%', gender='Ж')
            ],
            'mcv': [
                ReferenceRange(80, 100, 'фл')
            ],
            'mch': [
                ReferenceRange(27, 31, 'пг')
            ],
            'mchc': [
                ReferenceRange(320, 360, 'г/л')
            ],
            
            # Биохимический анализ крови
            'glucose': [
                ReferenceRange(3.3, 5.5, 'ммоль/л'),
                ReferenceRange(59, 99, 'мг/дл')
            ],
            'cholesterol_total': [
                ReferenceRange(0, 5.2, 'ммоль/л'),
                ReferenceRange(0, 200, 'мг/дл')
            ],
            'cholesterol_hdl': [
                ReferenceRange(1.0, 999, 'ммоль/л', gender='М'),
                ReferenceRange(1.2, 999, 'ммоль/л', gender='Ж')
            ],
            'cholesterol_ldl': [
                ReferenceRange(0, 3.0, 'ммоль/л'),
                ReferenceRange(0, 115, 'мг/дл')
            ],
            'triglycerides': [
                ReferenceRange(0, 1.7, 'ммоль/л'),
                ReferenceRange(0, 150, 'мг/дл')
            ],
            'bilirubin_total': [
                ReferenceRange(8.5, 20.5, 'мкмоль/л'),
                ReferenceRange(0.5, 1.2, 'мг/дл')
            ],
            'bilirubin_direct': [
                ReferenceRange(0, 8.6, 'мкмоль/л'),
                ReferenceRange(0, 0.5, 'мг/дл')
            ],
            'alt': [
                ReferenceRange(0, 40, 'ед/л', gender='М'),
                ReferenceRange(0, 32, 'ед/л', gender='Ж')
            ],
            'ast': [
                ReferenceRange(0, 40, 'ед/л', gender='М'),
                ReferenceRange(0, 32, 'ед/л', gender='Ж')
            ],
            'creatinine': [
                ReferenceRange(62, 115, 'мкмоль/л', gender='М'),
                ReferenceRange(53, 97, 'мкмоль/л', gender='Ж'),
                ReferenceRange(0.7, 1.3, 'мг/дл', gender='М'),
                ReferenceRange(0.6, 1.1, 'мг/дл', gender='Ж')
            ],
            'urea': [
                ReferenceRange(2.5, 7.5, 'ммоль/л'),
                ReferenceRange(15, 45, 'мг/дл')
            ],
            'protein_total': [
                ReferenceRange(64, 87, 'г/л'),
                ReferenceRange(6.4, 8.7, 'г/дл')
            ],
            'albumin': [
                ReferenceRange(35, 52, 'г/л'),
                ReferenceRange(3.5, 5.2, 'г/дл')
            ],
            
            # Гормоны
            'tsh': [
                ReferenceRange(0.27, 4.2, 'мЕд/л')
            ],
            't4_free': [
                ReferenceRange(12, 22, 'пмоль/л'),
                ReferenceRange(0.93, 1.7, 'нг/дл')
            ],
            't3_free': [
                ReferenceRange(3.1, 6.8, 'пмоль/л'),
                ReferenceRange(2.0, 4.4, 'пг/мл')
            ],
            
            # Маркеры воспаления
            'esr': [
                ReferenceRange(0, 15, 'мм/ч', gender='М'),
                ReferenceRange(0, 20, 'мм/ч', gender='Ж')
            ],
            'crp': [
                ReferenceRange(0, 3.0, 'мг/л'),
                ReferenceRange(0, 0.3, 'мг/дл')
            ],
            
            # Коагулограмма
            'pt': [
                ReferenceRange(11, 15, 'сек')
            ],
            'inr': [
                ReferenceRange(0.85, 1.15, '')
            ],
            'aptt': [
                ReferenceRange(25, 35, 'сек')
            ],
            'fibrinogen': [
                ReferenceRange(2.0, 4.0, 'г/л'),
                ReferenceRange(200, 400, 'мг/дл')
            ]
        }
        
        return ranges
    
    def _load_parameter_synonyms(self) -> Dict[str, List[str]]:
        """Загрузка синонимов параметров"""
        synonyms = {
            'hemoglobin': ['гемоглобин', 'hb', 'hgb', 'гб'],
            'erythrocytes': ['эритроциты', 'rbc', 'red blood cells', 'эр'],
            'leukocytes': ['лейкоциты', 'wbc', 'white blood cells', 'лейк'],
            'platelets': ['тромбоциты', 'plt', 'platelets', 'тромб'],
            'hematocrit': ['гематокрит', 'hct', 'гкт'],
            'glucose': ['глюкоза', 'glu', 'glucose', 'сахар'],
            'cholesterol_total': ['холестерин общий', 'total cholesterol', 'chol', 'хол'],
            'cholesterol_hdl': ['холестерин лпвп', 'hdl', 'лпвп'],
            'cholesterol_ldl': ['холестерин лпнп', 'ldl', 'лпнп'],
            'triglycerides': ['триглицериды', 'tg', 'триг'],
            'bilirubin_total': ['билирубин общий', 'total bilirubin', 'tbil'],
            'bilirubin_direct': ['билирубин прямой', 'direct bilirubin', 'dbil'],
            'alt': ['алт', 'alanine aminotransferase', 'алат'],
            'ast': ['аст', 'aspartate aminotransferase', 'асат'],
            'creatinine': ['креатинин', 'crea', 'cr'],
            'urea': ['мочевина', 'urea', 'bun'],
            'protein_total': ['белок общий', 'total protein', 'tp'],
            'albumin': ['альбумин', 'alb'],
            'tsh': ['ттг', 'thyroid stimulating hormone'],
            't4_free': ['т4 свободный', 'free t4', 'ft4'],
            't3_free': ['т3 свободный', 'free t3', 'ft3'],
            'esr': ['соэ', 'скорость оседания эритроцитов'],
            'crp': ['срб', 'c-reactive protein', 'с-реактивный белок'],
            'pt': ['протромбиновое время', 'prothrombin time'],
            'inr': ['мно', 'international normalized ratio'],
            'aptt': ['ачтв', 'activated partial thromboplastin time'],
            'fibrinogen': ['фибриноген', 'fib']
        }
        
        return synonyms
    
    def normalize_parameter_name(self, parameter: str) -> str:
        """Нормализация названия параметра"""
        parameter_lower = parameter.lower().strip()
        
        # Удаление лишних символов
        parameter_clean = re.sub(r'[^\w\s]', '', parameter_lower)
        
        # Поиск в синонимах
        for standard_name, synonyms in self.parameter_synonyms.items():
            if parameter_clean in [syn.lower() for syn in synonyms]:
                return standard_name
            
            # Поиск частичного совпадения
            for synonym in synonyms:
                if synonym.lower() in parameter_clean or parameter_clean in synonym.lower():
                    return standard_name
        
        return parameter_clean
    
    def extract_from_pdf(self, pdf_file) -> List[LabResult]:
        """Извлечение данных из PDF файла"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            return self._parse_text_data(full_text)
        
        except Exception as e:
            raise Exception(f"Ошибка при обработке PDF: {e}")
    
    def extract_from_excel(self, excel_file) -> List[LabResult]:
        """Извлечение данных из Excel файла"""
        try:
            # Попытка прочитать все листы
            sheets = pd.read_excel(excel_file, sheet_name=None)
            all_results = []
            
            for sheet_name, df in sheets.items():
                results = self._parse_dataframe(df)
                all_results.extend(results)
            
            return all_results
        
        except Exception as e:
            raise Exception(f"Ошибка при обработке Excel: {e}")
    
    def extract_from_csv(self, csv_file) -> List[LabResult]:
        """Извлечение данных из CSV файла"""
        try:
            # Попытка определить разделитель
            sample = csv_file.read(1024).decode('utf-8')
            csv_file.seek(0)
            
            delimiter = ','
            if ';' in sample:
                delimiter = ';'
            elif '\t' in sample:
                delimiter = '\t'
            
            df = pd.read_csv(csv_file, delimiter=delimiter)
            return self._parse_dataframe(df)
        
        except Exception as e:
            raise Exception(f"Ошибка при обработке CSV: {e}")
    
    def extract_from_json(self, json_file) -> List[LabResult]:
        """Извлечение данных из JSON файла"""
        try:
            data = json.load(json_file)
            
            if isinstance(data, list):
                # Массив результатов
                return self._parse_json_array(data)
            elif isinstance(data, dict):
                # Объект с результатами
                return self._parse_json_object(data)
            else:
                raise Exception("Неподдерживаемая структура JSON")
        
        except Exception as e:
            raise Exception(f"Ошибка при обработке JSON: {e}")
    
    def extract_from_xml(self, xml_file) -> List[LabResult]:
        """Извлечение данных из XML файла"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            results = []
            
            # Поиск различных структур XML
            for result_elem in root.findall('.//result') + root.findall('.//test') + root.findall('.//parameter'):
                result = self._parse_xml_element(result_elem)
                if result:
                    results.append(result)
            
            return results
        
        except Exception as e:
            raise Exception(f"Ошибка при обработке XML: {e}")
    
    def _parse_text_data(self, text: str) -> List[LabResult]:
        """Парсинг текстовых данных"""
        results = []
        
        # Различные паттерны для извлечения данных
        patterns = [
            r'([А-Яа-я\w\s]+):\s*(\d+[.,]\d+|\d+)\s*([А-Яа-я\w/\s\^]*)',
            r'([А-Яа-я\w\s]+)\s+(\d+[.,]\d+|\d+)\s+([А-Яа-я\w/\s\^]*)',
            r'([А-Яа-я\w\s]+)\s*-\s*(\d+[.,]\d+|\d+)\s*([А-Яа-я\w/\s\^]*)',
            r'([A-Za-z\w\s]+):\s*(\d+[.,]\d+|\d+)\s*([A-Za-z\w/\s\^]*)',
            r'([A-Za-z\w\s]+)\s+(\d+[.,]\d+|\d+)\s+([A-Za-z\w/\s\^]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            
            for match in matches:
                parameter = match[0].strip()
                value_str = match[1].replace(',', '.')
                unit = match[2].strip()
                
                # Фильтрация нежелательных совпадений
                if len(parameter) < 3 or parameter.lower() in ['дата', 'время', 'page', 'результат']:
                    continue
                
                try:
                    value = float(value_str)
                    normalized_param = self.normalize_parameter_name(parameter)
                    
                    lab_result = LabResult(
                        parameter=parameter,
                        value=value,
                        unit=unit
                    )
                    
                    # Добавление референсного диапазона
                    self._add_reference_range(lab_result, normalized_param)
                    
                    results.append(lab_result)
                
                except ValueError:
                    continue
        
        return results
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[LabResult]:
        """Парсинг данных из DataFrame"""
        results = []
        
        # Возможные названия колонок
        param_cols = ['параметр', 'показатель', 'анализ', 'parameter', 'test', 'name']
        value_cols = ['значение', 'результат', 'value', 'result']
        unit_cols = ['единица', 'ед', 'unit', 'units']
        ref_cols = ['норма', 'референс', 'reference', 'ref_range']
        
        # Поиск подходящих колонок
        param_col = self._find_column(df, param_cols)
        value_col = self._find_column(df, value_cols)
        unit_col = self._find_column(df, unit_cols)
        ref_col = self._find_column(df, ref_cols)
        
        if not param_col or not value_col:
            # Попытка автоматического определения структуры
            return self._auto_parse_dataframe(df)
        
        for idx, row in df.iterrows():
            if pd.notna(row[param_col]) and pd.notna(row[value_col]):
                try:
                    parameter = str(row[param_col]).strip()
                    value = float(str(row[value_col]).replace(',', '.'))
                    unit = str(row[unit_col]).strip() if unit_col and pd.notna(row[unit_col]) else ""
                    
                    normalized_param = self.normalize_parameter_name(parameter)
                    
                    lab_result = LabResult(
                        parameter=parameter,
                        value=value,
                        unit=unit
                    )
                    
                    # Добавление референсного диапазона
                    if ref_col and pd.notna(row[ref_col]):
                        ref_range = self._parse_reference_range(str(row[ref_col]), unit)
                        if ref_range:
                            lab_result.reference_range = ref_range
                    else:
                        self._add_reference_range(lab_result, normalized_param)
                    
                    results.append(lab_result)
                
                except (ValueError, TypeError):
                    continue
        
        return results
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Поиск колонки по возможным названиям"""
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for name in possible_names:
                if name in col_lower:
                    return col
        return None
    
    def _auto_parse_dataframe(self, df: pd.DataFrame) -> List[LabResult]:
        """Автоматический парсинг DataFrame"""
        results = []
        
        # Если DataFrame имеет 2 колонки, предполагаем parameter-value
        if len(df.columns) == 2:
            param_col, value_col = df.columns[0], df.columns[1]
            
            for idx, row in df.iterrows():
                if pd.notna(row[param_col]) and pd.notna(row[value_col]):
                    try:
                        parameter = str(row[param_col]).strip()
                        value = float(str(row[value_col]).replace(',', '.'))
                        
                        normalized_param = self.normalize_parameter_name(parameter)
                        
                        lab_result = LabResult(
                            parameter=parameter,
                            value=value,
                            unit=""
                        )
                        
                        self._add_reference_range(lab_result, normalized_param)
                        results.append(lab_result)
                    
                    except (ValueError, TypeError):
                        continue
        
        return results
    
    def _parse_json_array(self, data: List) -> List[LabResult]:
        """Парсинг массива JSON"""
        results = []
        
        for item in data:
            if isinstance(item, dict):
                result = self._parse_json_item(item)
                if result:
                    results.append(result)
        
        return results
    
    def _parse_json_object(self, data: Dict) -> List[LabResult]:
        """Парсинг объекта JSON"""
        results = []
        
        # Поиск массива результатов
        if 'results' in data:
            return self._parse_json_array(data['results'])
        elif 'tests' in data:
            return self._parse_json_array(data['tests'])
        elif 'parameters' in data:
            return self._parse_json_array(data['parameters'])
        else:
            # Прямой парсинг как ключ-значение
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    normalized_param = self.normalize_parameter_name(key)
                    
                    lab_result = LabResult(
                        parameter=key,
                        value=float(value),
                        unit=""
                    )
                    
                    self._add_reference_range(lab_result, normalized_param)
                    results.append(lab_result)
        
        return results
    
    def _parse_json_item(self, item: Dict) -> Optional[LabResult]:
        """Парсинг элемента JSON"""
        # Возможные ключи для параметра
        param_keys = ['parameter', 'name', 'test', 'параметр', 'показатель']
        # Возможные ключи для значения
        value_keys = ['value', 'result', 'значение', 'результат']
        # Возможные ключи для единиц
        unit_keys = ['unit', 'units', 'единица', 'ед']
        
        parameter = None
        value = None
        unit = ""
        
        # Поиск параметра
        for key in param_keys:
            if key in item:
                parameter = str(item[key])
                break
        
        # Поиск значения
        for key in value_keys:
            if key in item:
                try:
                    value = float(item[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Поиск единиц
        for key in unit_keys:
            if key in item:
                unit = str(item[key])
                break
        
        if parameter and value is not None:
            normalized_param = self.normalize_parameter_name(parameter)
            
            lab_result = LabResult(
                parameter=parameter,
                value=value,
                unit=unit
            )
            
            self._add_reference_range(lab_result, normalized_param)
            return lab_result
        
        return None
    
    def _parse_xml_element(self, element: ET.Element) -> Optional[LabResult]:
        """Парсинг XML элемента"""
        parameter = None
        value = None
        unit = ""
        
        # Поиск в атрибутах
        if 'name' in element.attrib:
            parameter = element.attrib['name']
        elif 'parameter' in element.attrib:
            parameter = element.attrib['parameter']
        
        if 'value' in element.attrib:
            try:
                value = float(element.attrib['value'])
            except ValueError:
                pass
        
        if 'unit' in element.attrib:
            unit = element.attrib['unit']
        
        # Поиск в дочерних элементах
        if parameter is None:
            name_elem = element.find('name') or element.find('parameter')
            if name_elem is not None:
                parameter = name_elem.text
        
        if value is None:
            value_elem = element.find('value') or element.find('result')
            if value_elem is not None:
                try:
                    value = float(value_elem.text)
                except (ValueError, TypeError):
                    pass
        
        if not unit:
            unit_elem = element.find('unit') or element.find('units')
            if unit_elem is not None:
                unit = unit_elem.text or ""
        
        if parameter and value is not None:
            normalized_param = self.normalize_parameter_name(parameter)
            
            lab_result = LabResult(
                parameter=parameter,
                value=value,
                unit=unit
            )
            
            self._add_reference_range(lab_result, normalized_param)
            return lab_result
        
        return None
    
    def _add_reference_range(self, lab_result: LabResult, normalized_param: str):
        """Добавление референсного диапазона"""
        if normalized_param in self.reference_ranges:
            ranges = self.reference_ranges[normalized_param]
            
            # Выбор подходящего диапазона (упрощенная логика)
            best_range = None
            for ref_range in ranges:
                # Проверка единиц измерения
                if self._units_match(lab_result.unit, ref_range.unit):
                    best_range = ref_range
                    break
            
            if best_range:
                lab_result.reference_range = best_range
                lab_result.status = self._determine_status(lab_result.value, best_range)
    
    def _units_match(self, unit1: str, unit2: str) -> bool:
        """Проверка соответствия единиц измерения"""
        unit1_clean = re.sub(r'[^\w]', '', unit1.lower())
        unit2_clean = re.sub(r'[^\w]', '', unit2.lower())
        
        return unit1_clean == unit2_clean or unit1_clean in unit2_clean or unit2_clean in unit1_clean
    
    def _determine_status(self, value: float, ref_range: ReferenceRange) -> str:
        """Определение статуса результата"""
        if value < ref_range.min_value:
            return "Ниже нормы"
        elif value > ref_range.max_value:
            return "Выше нормы"
        else:
            return "Норма"
    
    def _parse_reference_range(self, ref_text: str, unit: str) -> Optional[ReferenceRange]:
        """Парсинг текста референсного диапазона"""
        # Паттерны для различных форматов
        patterns = [
            r'(\d+[.,]\d*)\s*-\s*(\d+[.,]\d*)',
            r'(\d+[.,]\d*)\s*до\s*(\d+[.,]\d*)',
            r'(\d+[.,]\d*)\s*to\s*(\d+[.,]\d*)',
            r'<\s*(\d+[.,]\d*)',  # только верхняя граница
            r'>\s*(\d+[.,]\d*)'   # только нижняя граница
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ref_text)
            if match:
                if len(match.groups()) == 2:
                    min_val = float(match.group(1).replace(',', '.'))
                    max_val = float(match.group(2).replace(',', '.'))
                    return ReferenceRange(min_val, max_val, unit)
                elif '<' in pattern:
                    max_val = float(match.group(1).replace(',', '.'))
                    return ReferenceRange(0, max_val, unit)
                elif '>' in pattern:
                    min_val = float(match.group(1).replace(',', '.'))
                    return ReferenceRange(min_val, 999999, unit)
        
        return None
    
    def analyze_results(self, results: List[LabResult]) -> Dict:
        """Анализ результатов лабораторных исследований"""
        analysis = {
            'total_parameters': len(results),
            'normal_count': 0,
            'abnormal_count': 0,
            'above_normal': 0,
            'below_normal': 0,
            'unknown_status': 0,
            'categories': {},
            'critical_values': [],
            'recommendations': []
        }
        
        # Категоризация параметров
        categories = {
            'Общий анализ крови': ['hemoglobin', 'erythrocytes', 'leukocytes', 'platelets', 'hematocrit'],
            'Биохимия крови': ['glucose', 'cholesterol_total', 'bilirubin_total', 'alt', 'ast', 'creatinine', 'urea'],
            'Липидный профиль': ['cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides'],
            'Функция печени': ['alt', 'ast', 'bilirubin_total', 'bilirubin_direct', 'albumin'],
            'Функция почек': ['creatinine', 'urea'],
            'Гормоны': ['tsh', 't4_free', 't3_free'],
            'Коагулограмма': ['pt', 'inr', 'aptt', 'fibrinogen']
        }
        
        for result in results:
            # Подсчет статусов
            if result.status == "Норма":
                analysis['normal_count'] += 1
            elif result.status == "Выше нормы":
                analysis['abnormal_count'] += 1
                analysis['above_normal'] += 1
            elif result.status == "Ниже нормы":
                analysis['abnormal_count'] += 1
                analysis['below_normal'] += 1
            else:
                analysis['unknown_status'] += 1
            
            # Категоризация
            normalized_param = self.normalize_parameter_name(result.parameter)
            for category, params in categories.items():
                if normalized_param in params:
                    if category not in analysis['categories']:
                        analysis['categories'][category] = {'total': 0, 'normal': 0, 'abnormal': 0}
                    
                    analysis['categories'][category]['total'] += 1
                    if result.status == "Норма":
                        analysis['categories'][category]['normal'] += 1
                    elif result.status in ["Выше нормы", "Ниже нормы"]:
                        analysis['categories'][category]['abnormal'] += 1
            
            # Выявление критических значений
            if self._is_critical_value(result, normalized_param):
                analysis['critical_values'].append({
                    'parameter': result.parameter,
                    'value': result.value,
                    'unit': result.unit,
                    'status': result.status,
                    'severity': self._assess_severity(result, normalized_param)
                })
        
        # Генерация рекомендаций
        analysis['recommendations'] = self._generate_recommendations(results, analysis)
        
        return analysis
    
    def _is_critical_value(self, result: LabResult, normalized_param: str) -> bool:
        """Определение критических значений"""
        critical_thresholds = {
            'glucose': {'low': 2.8, 'high': 15.0},
            'hemoglobin': {'low': 80, 'high': 200},
            'leukocytes': {'low': 2.0, 'high': 20.0},
            'platelets': {'low': 100, 'high': 1000},
            'creatinine': {'low': 0, 'high': 300},  # мкмоль/л
            'bilirubin_total': {'low': 0, 'high': 100},
            'alt': {'low': 0, 'high': 200},
            'ast': {'low': 0, 'high': 200}
        }
        
        if normalized_param in critical_thresholds:
            thresholds = critical_thresholds[normalized_param]
            return result.value <= thresholds['low'] or result.value >= thresholds['high']
        
        return False
    
    def _assess_severity(self, result: LabResult, normalized_param: str) -> str:
        """Оценка тяжести отклонения"""
        if not result.reference_range:
            return "Неопределенная"
        
        ref_range = result.reference_range
        range_width = ref_range.max_value - ref_range.min_value
        
        if result.value < ref_range.min_value:
            deviation = (ref_range.min_value - result.value) / range_width
        elif result.value > ref_range.max_value:
            deviation = (result.value - ref_range.max_value) / range_width
        else:
            return "Норма"
        
        if deviation < 0.5:
            return "Легкая"
        elif deviation < 1.0:
            return "Умеренная"
        else:
            return "Выраженная"
    
    def _generate_recommendations(self, results: List[LabResult], analysis: Dict) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации по критическим значениям
        if analysis['critical_values']:
            recommendations.append("Обнаружены критические отклонения - требуется срочная консультация врача")
        
        # Рекомендации по категориям
        for category, stats in analysis['categories'].items():
            if stats['abnormal'] > stats['normal']:
                if category == "Функция печени":
                    recommendations.append("Рекомендуется консультация гастроэнтеролога/гепатолога")
                elif category == "Функция почек":
                    recommendations.append("Рекомендуется консультация нефролога")
                elif category == "Гормоны":
                    recommendations.append("Рекомендуется консультация эндокринолога")
                elif category == "Коагулограмма":
                    recommendations.append("Рекомендуется консультация гематолога")
        
        # Общие рекомендации
        if analysis['abnormal_count'] == 0:
            recommendations.append("Все показатели в пределах нормы - рекомендуется плановое наблюдение")
        elif analysis['abnormal_count'] < analysis['total_parameters'] * 0.3:
            recommendations.append("Выявлены незначительные отклонения - рекомендуется динамическое наблюдение")
        
        if not recommendations:
            recommendations.append("Рекомендуется интерпретация результатов лечащим врачом")
        
        return recommendations
    
    def generate_report(self, results: List[LabResult], patient_info: Dict = None) -> Dict:
        """Генерация полного отчета"""
        analysis = self.analyze_results(results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'patient_info': patient_info or {},
            'summary': {
                'total_tests': analysis['total_parameters'],
                'normal_results': analysis['normal_count'],
                'abnormal_results': analysis['abnormal_count'],
                'completion_rate': f"{(analysis['normal_count'] + analysis['abnormal_count']) / analysis['total_parameters'] * 100:.1f}%" if analysis['total_parameters'] > 0 else "0%"
            },
            'detailed_results': [
                {
                    'parameter': result.parameter,
                    'value': result.value,
                    'unit': result.unit,
                    'reference_range': f"{result.reference_range.min_value}-{result.reference_range.max_value} {result.reference_range.unit}" if result.reference_range else "Не определено",
                    'status': result.status,
                    'comment': result.comment
                } for result in results
            ],
            'category_analysis': analysis['categories'],
            'critical_findings': analysis['critical_values'],
            'recommendations': analysis['recommendations'],
            'quality_metrics': {
                'parameters_with_references': sum(1 for r in results if r.reference_range),
                'coverage_percentage': sum(1 for r in results if r.reference_range) / len(results) * 100 if results else 0
            }
        }
        
        return report