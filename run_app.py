#!/usr/bin/env python3
"""
Скрипт для запуска Медицинского Ассистента с ИИ

Автоматически проверяет зависимости, настройки и запускает приложение.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Ошибка: Требуется Python 3.8 или новее")
        print(f"Текущая версия: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python версия: {sys.version.split()[0]}")

def check_requirements():
    """Проверка установленных зависимостей"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ Файл requirements.txt не найден")
        return False
    
    print("🔍 Проверка зависимостей...")
    
    missing_packages = []
    
    # Основные зависимости
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "PIL",
        "anthropic"
    ]
    
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL")
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("Запустите: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Проверка переменных окружения"""
    print("\n🔧 Проверка конфигурации...")
    
    # Проверка файла .env
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  Файл .env не найден")
            print("📄 Создайте .env на основе .env.example")
        else:
            print("ℹ️  Файлы конфигурации не найдены")
        print("🤖 ИИ-функции могут быть недоступны")
        return False
    
    # Загрузка переменных окружения
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            print("✅ API ключ Anthropic найден")
            return True
        else:
            print("⚠️  API ключ Anthropic не настроен")
            print("🤖 ИИ-функции будут недоступны")
            return False
            
    except ImportError:
        print("⚠️  python-dotenv не установлен")
        print("Запустите: pip install python-dotenv")
        return False

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        "data",
        "uploads", 
        "temp",
        "backups",
        "logs",
        ".streamlit"
    ]
    
    print("\n📁 Создание директорий...")
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Создана: {directory}")
        else:
            print(f"ℹ️  Существует: {directory}")

def check_streamlit_config():
    """Проверка конфигурации Streamlit"""
    config_file = Path(".streamlit/config.toml")
    
    if not config_file.exists():
        print("📝 Создание конфигурации Streamlit...")
        
        config_content = """[global]
dataFrameSerialization = "legacy"

[server]
runOnSave = true
port = 8501
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("✅ Конфигурация Streamlit создана")
    else:
        print("✅ Конфигурация Streamlit найдена")

def install_requirements():
    """Установка зависимостей"""
    print("\n📦 Установка зависимостей...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка установки зависимостей")
        return False

def run_application():
    """Запуск приложения"""
    print("\n🚀 Запуск Медицинского Ассистента...")
    print("🌐 Приложение будет доступно по адресу: http://localhost:8501")
    print("⏹️  Для остановки нажмите Ctrl+C")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.fileWatcherType", "auto"
        ])
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено")
    except FileNotFoundError:
        print("❌ Streamlit не найден. Установите: pip install streamlit")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

def show_help():
    """Показать справку"""
    help_text = """
🏥 МЕДИЦИНСКИЙ АССИСТЕНТ С ИИ

Использование:
    python run_app.py [опции]

Опции:
    --help, -h          Показать эту справку
    --check             Только проверить зависимости
    --install           Установить зависимости
    --setup             Полная настройка
    --dev               Режим разработки
    --port PORT         Указать порт (по умолчанию 8501)

Примеры:
    python run_app.py                # Обычный запуск
    python run_app.py --check        # Проверка системы
    python run_app.py --setup        # Первоначальная настройка
    python run_app.py --port 8080    # Запуск на порту 8080

Требования:
    - Python 3.8+
    - Интернет-соединение для установки пакетов
    - API ключ Anthropic (для ИИ-функций)

Конфигурация:
    1. Скопируйте .env.example в .env
    2. Добавьте ваш API ключ Anthropic
    3. Настройте другие параметры по необходимости

Поддержка:
    GitHub: https://github.com/vasiliys961/medical-assistant
    Email: vasiliys961@example.com
"""
    print(help_text)

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Медицинский Ассистент с ИИ", add_help=False)
    parser.add_argument("--help", "-h", action="store_true", help="Показать справку")
    parser.add_argument("--check", action="store_true", help="Проверить зависимости")
    parser.add_argument("--install", action="store_true", help="Установить зависимости")
    parser.add_argument("--setup", action="store_true", help="Полная настройка")
    parser.add_argument("--dev", action="store_true", help="Режим разработки")
    parser.add_argument("--port", type=int, default=8501, help="Порт для запуска")
    
    args = parser.parse_args()
    
    # Заголовок
    print("=" * 60)
    print("🏥 МЕДИЦИНСКИЙ АССИСТЕНТ С ИСКУССТВЕННЫМ ИНТЕЛЛЕКТОМ")
    print("=" * 60)
    
    if args.help:
        show_help()
        return
    
    # Проверка версии Python
    check_python_version()
    
    if args.check:
        print("\n🔍 РЕЖИМ ПРОВЕРКИ")
        check_requirements()
        check_environment()
        return
    
    if args.install:
        print("\n📦 РЕЖИМ УСТАНОВКИ")
        install_requirements()
        return
    
    if args.setup:
        print("\n⚙️ РЕЖИМ НАСТРОЙКИ")
        create_directories()
        check_streamlit_config()
        
        if not check_requirements():
            print("\n📦 Установка недостающих зависимостей...")
            if not install_requirements():
                print("❌ Не удалось установить зависимости")
                return
        
        check_environment()
        print("\n✅ Настройка завершена!")
        
        if input("\n🚀 Запустить приложение? (y/n): ").lower() == 'y':
            run_application()
        return
    
    # Обычный запуск
    print("\n🔧 Предварительная проверка...")
    
    # Создание директорий
    create_directories()
    
    # Проверка конфигурации Streamlit
    check_streamlit_config()
    
    # Проверка зависимостей
    if not check_requirements():
        if input("\n📦 Установить отсутствующие зависимости? (y/n): ").lower() == 'y':
            if not install_requirements():
                print("❌ Не удалось установить зависимости")
                return
        else:
            print("⚠️ Приложение может работать некорректно")
    
    # Проверка переменных окружения
    env_ok = check_environment()
    if not env_ok:
        print("\n⚠️ Некоторые функции ИИ могут быть недоступны")
        if input("Продолжить запуск? (y/n): ").lower() != 'y':
            return
    
    # Запуск приложения
    if args.dev:
        print("\n🔧 РЕЖИМ РАЗРАБОТКИ")
        os.environ['STREAMLIT_DEV_MODE'] = 'true'
    
    if args.port != 8501:
        os.environ['STREAMLIT_SERVER_PORT'] = str(args.port)
    
    run_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Выход из программы")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("📧 Сообщите об ошибке: vasiliys961@example.com")