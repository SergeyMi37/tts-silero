import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import os
import threading
import pygame
import numpy as np
import logging
from datetime import datetime
import sys
import json
import hashlib
import subprocess
import re
import wave
from pydub import AudioSegment
from text_preprocessor import TextPreprocessor
import html
import argparse

# Константы для логирования
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'

# Настройка логирования (только в консоль)
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Константы
SAMPLE_RATE = 48000
MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_ru.pt'
MODEL_NAME = 'v5_ru.pt'
CONFIG_FILE = 'text2mp3.json'

# Путь к кэшу модели в LOCALAPPDATA
CACHE_DIR = os.path.join(
    os.getenv('LOCALAPPDATA', os.path.expanduser('~')),
    'SileroTTS', 'models'
)
MODEL_FILE = os.path.join(CACHE_DIR, MODEL_NAME)

# Путь к директории для сохранения аудио файлов
AUDIO_DIR = os.path.join(
    os.getenv('LOCALAPPDATA', os.path.expanduser('~')),
    'SileroTTS', 'audiofiles'
)

# Доступные русские голоса из Silero v5
SPEAKERS = ['baya', 'eugene', 'kseniya', 'xenia', 'random']

# Настройки разбиения на кусочки больших текстов
AUTO_CHUNK_THRESHOLD = 3000
DEFAULT_MAX_CHARS_PER_CHUNK = 1200
DEFAULT_SILENCE_MS = 200

# Тестовый текст по умолчанию
DEFAULT_DEMO_TEXT = "Это демонстрация работы Silero TTS версии 5. Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности! В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров."

class TextHandler(logging.Handler):
    """Кастомный обработчик логов для вывода в текстовое поле Tkinter"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record):
        try:
            msg = self.format(record)
            def append():
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)  # Автопрокрутка к последнему сообщению
            # Безопасный вызов в главном потоке
            if hasattr(self.text_widget, 'after'):
                self.text_widget.after(0, append)
            else:
                append()
        except Exception:
            self.handleError(record)


class SileroTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Silero TTS Озвучка текста (v5)")
        self.root.geometry("1200x500")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.model = None
        self.is_model_loaded = False
        self.current_sound = None  # Для хранения текущего звукового объекта
        self.demo_text = DEFAULT_DEMO_TEXT  # Тестовый текст по умолчанию
        self.stop_generation_flag = False  # Флаг остановки генерации
        self.last_loaded_file_path = None  # Путь к последнему загруженному файлу
        
        # Переменные настроек предобработки текста
        self.use_preprocessing_var = tk.BooleanVar(value=False)
        self.use_num2words_var = tk.BooleanVar(value=True)
        self.use_ruaccent_var = tk.BooleanVar(value=False)
        
        # Инициализация препроцессора текста
        self.text_preprocessor = TextPreprocessor()
        self.preprocessor_loaded = False
        
        logging.info("Инициализация приложения SileroTTSApp")
        
        try:
            # Инициализация pygame mixer (channels=2 для стерео)
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
            logging.info(f"Pygame mixer инициализирован: частота={SAMPLE_RATE}Hz, каналы=2")
        except Exception as e:
            logging.error(f"Ошибка инициализации pygame mixer: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось инициализировать звуковую систему: {e}")
            sys.exit(1)
        
        self.load_config()
        self.setup_ui()
        self.apply_saved_config()
        self.load_model_threaded()
    
    def load_config(self):
        """Загрузка настроек из JSON файла"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.saved_config = config
                logging.info(f"Настройки загружены из {CONFIG_FILE}")
            else:
                self.saved_config = {}
                logging.info("Файл настроек не найден, будут использованы значения по умолчанию")
        except Exception as e:
            logging.error(f"Ошибка при загрузке настроек: {e}")
            self.saved_config = {}
    
    def add_pause_tags(self, text):
        """Добавление тегов пауз <s> после точек в конце предложений.
        
        Аргументы:
            text: Текст для обработки
        
        Возвращает:
            Текст с добавленными тегами пауз после точек (валидный XML)
        """
        if not text:
            return text
        
        # Добавляем теги пауз <s> вокруг знаков препинания для валидного XML
        # Заменяем . на <s>.</s>, ? на <s>?</s>, ! на <s>!</s>
        result = re.sub(r'\.\s*', '<s>.</s> ', text)
        result = re.sub(r'\?\s*', '<s>?</s> ', result)
        result = re.sub(r'!\s*', '<s>!</s> ', result)
        result = re.sub(r'…\s*', '<s>…</s> ', result)
        
        # Удаляем множественные пробелы
        result = ' '.join(result.split())
        
        logging.debug(f"Добавлены теги пауз: '{text[:50]}...' → '{result[:50]}...'")
        return result
    
    def _validate_geometry(self, geometry_str):
        """Валидация строки геометрии окна.
        
        Проверяет, что ширина и высота окна разумные (>= 200x150).
        Возвращает корректную строку геометрии или значение по умолчанию.
        """
        try:
            # Парсинг строки геометрии в формате "WIDTHxHEIGHT+X+Y"
            match = re.match(r'(\d+)x(\d+)([+-]\d+)?([+-]\d+)?', geometry_str)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                
                # Проверяем минимальные размеры
                if width < 200 or height < 150:
                    logging.warning(f"Некорректная геометрия окна ({width}x{height}), используется значение по умолчанию")
                    return '1200x500'
                
                # Возвращаем оригинальную строку если всё корректно
                return geometry_str
            else:
                logging.warning(f"Не удалось распарсить геометрию: {geometry_str}")
                return '1200x500'
        except Exception as e:
            logging.error(f"Ошибка валидации геометрии: {e}")
            return '1200x500'
    
    def save_config(self):
        """Сохранение настроек в JSON файл"""
        try:
            # Получаем текущую геометрию и валидируем её
            current_geometry = self.root.geometry() if hasattr(self, 'root') else '600x500'
            validated_geometry = self._validate_geometry(current_geometry)
            
            config = {
                'speaker': self.speaker_combo.get() if hasattr(self, 'speaker_combo') else SPEAKERS[0],
                'text': self.text_area.get("1.0", tk.END).strip() if hasattr(self, 'text_area') else '',
                'window_geometry': validated_geometry,
                'chunk_mode': bool(self.chunk_mode_var.get()) if hasattr(self, 'chunk_mode_var') else False,
                'save_parts': bool(self.save_parts_var.get()) if hasattr(self, 'save_parts_var') else False,
                'max_chars_per_chunk': int(self.max_chars_var.get()) if hasattr(self, 'max_chars_var') else DEFAULT_MAX_CHARS_PER_CHUNK,
                'silence_ms': int(self.silence_ms_var.get()) if hasattr(self, 'silence_ms_var') else DEFAULT_SILENCE_MS,
                'chunk_dir': self.chunk_dir_var.get() if hasattr(self, 'chunk_dir_var') else 'my_audiobook',
                'convert_to_mp3': bool(self.convert_to_mp3_var.get()) if hasattr(self, 'convert_to_mp3_var') else False,
                'mp3_bitrate': self.mp3_bitrate_var.get() if hasattr(self, 'mp3_bitrate_var') else '192k',
                'speech_rate': self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium',
                'demo_text': self.demo_text if hasattr(self, 'demo_text') else DEFAULT_DEMO_TEXT,
                'target_dir': self.target_dir_var.get() if hasattr(self, 'target_dir_var') else AUDIO_DIR,
                'use_preprocessing': bool(self.use_preprocessing_var.get()) if hasattr(self, 'use_preprocessing_var') else False,
                'use_num2words': bool(self.use_num2words_var.get()) if hasattr(self, 'use_num2words_var') else True,
                'use_ruaccent': bool(self.use_ruaccent_var.get()) if hasattr(self, 'use_ruaccent_var') else False,
                'last_loaded_file': self.last_loaded_file_path if hasattr(self, 'last_loaded_file_path') else None,
                'delete_wav_dir': bool(self.delete_wav_dir_var.get()) if hasattr(self, 'delete_wav_dir_var') else False
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logging.info(f"Настройки сохранены в {CONFIG_FILE}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении настроек: {e}")
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        logging.info("Настройка пользовательского интерфейса")
        
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки загрузки тестового текста и файла
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопка выбора файла
        self.load_file_btn = ttk.Button(buttons_frame, text="📄 Загрузить файл (txt/fb2)", command=self.load_file)
        self.load_file_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Кнопка загрузки тестового текста
        self.load_demo_btn = ttk.Button(buttons_frame, text="📝 Загрузить тестовый текст", command=self.load_demo_text)
        self.load_demo_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Кнопка генерации CLI команды
        self.generate_cli_btn = ttk.Button(buttons_frame, text="📋 Создать CLI команду", command=self.generate_cli_command)
        self.generate_cli_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Вкладки
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        text_tab = ttk.Frame(self.notebook)
        chunks_tab = ttk.Frame(self.notebook)
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(text_tab, text=" - Текст - ")
        self.notebook.add(chunks_tab, text=" - Кусочки - ")
        self.notebook.add(log_tab, text=" - Протокол - ")

        # --- Вкладка "Протокол" ---
        self.log_area = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        # Привязки клавиш для буфера обмена (только копирование, лог только для чтения)
        self.log_area.bind('<Control-c>', lambda e: self.copy_to_clipboard(self.log_area))
        self.log_area.bind('<Control-C>', lambda e: self.copy_to_clipboard(self.log_area))
        # Привязки клавиш для поиска
        self.log_area.bind('<Control-f>', lambda e: self.find_text(self.log_area))
        self.log_area.bind('<Control-F>', lambda e: self.find_text(self.log_area))
        self.log_area.bind('<F3>', lambda e: self.find_next(self.log_area))

        # --- Вкладка "Текст" ---
        ttk.Label(text_tab, text="Введите текст для озвучки:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))

        self.text_area = scrolledtext.ScrolledText(text_tab, wrap=tk.WORD, height=10, font=("Arial", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, DEFAULT_DEMO_TEXT)
        # Привязки клавиш для буфера обмена
        self.text_area.bind('<Control-c>', lambda e: self.copy_to_clipboard(self.text_area))
        self.text_area.bind('<Control-C>', lambda e: self.copy_to_clipboard(self.text_area))
        self.text_area.bind('<Control-v>', lambda e: self.paste_from_clipboard(self.text_area))
        self.text_area.bind('<Control-V>', lambda e: self.paste_from_clipboard(self.text_area))
        self.text_area.bind('<Control-x>', lambda e: self.cut_to_clipboard(self.text_area))
        self.text_area.bind('<Control-X>', lambda e: self.cut_to_clipboard(self.text_area))
        # Привязки клавиш для поиска
        self.text_area.bind('<Control-f>', lambda e: self.find_text(self.text_area))
        self.text_area.bind('<Control-F>', lambda e: self.find_text(self.text_area))
        self.text_area.bind('<F3>', lambda e: self.find_next(self.text_area))
        logging.debug(f"Текстовое поле инициализировано с текстом по умолчанию длиной {len(DEFAULT_DEMO_TEXT)} символов")

        # --- Вкладка "Кусочки" ---
        chunks_controls = ttk.Frame(chunks_tab)
        chunks_controls.pack(fill=tk.X, pady=(0, 5))

        self.split_chunks_btn = ttk.Button(chunks_controls, text="✂ Разделить текст на кусочки", command=self.split_text_to_chunks_ui)
        self.split_chunks_btn.pack(side=tk.LEFT)

        self.speak_chunks_btn = ttk.Button(chunks_controls, text="🔊 Озвучивать кусочки", command=self.speak_chunks_threaded)
        self.speak_chunks_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(chunks_controls, text="Имя директории:").pack(side=tk.LEFT, padx=(15, 5))
        self.chunk_dir_var = tk.StringVar(value="my_audiobook")
        self.chunk_dir_entry = ttk.Entry(chunks_controls, textvariable=self.chunk_dir_var, width=25)
        self.chunk_dir_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.chunks_area = scrolledtext.ScrolledText(chunks_tab, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.chunks_area.pack(fill=tk.BOTH, expand=True)
        # Привязки клавиш для буфера обмена
        self.chunks_area.bind('<Control-c>', lambda e: self.copy_to_clipboard(self.chunks_area))
        self.chunks_area.bind('<Control-C>', lambda e: self.copy_to_clipboard(self.chunks_area))
        self.chunks_area.bind('<Control-v>', lambda e: self.paste_from_clipboard(self.chunks_area))
        self.chunks_area.bind('<Control-V>', lambda e: self.paste_from_clipboard(self.chunks_area))
        self.chunks_area.bind('<Control-x>', lambda e: self.cut_to_clipboard(self.chunks_area))
        self.chunks_area.bind('<Control-X>', lambda e: self.cut_to_clipboard(self.chunks_area))
        # Привязки клавиш для поиска
        self.chunks_area.bind('<Control-f>', lambda e: self.find_text(self.chunks_area))
        self.chunks_area.bind('<Control-F>', lambda e: self.find_text(self.chunks_area))
        self.chunks_area.bind('<F3>', lambda e: self.find_next(self.chunks_area))
        
        # Фрейм для выбора голоса и кнопок
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Выбор голоса
        ttk.Label(controls_frame, text="Выберите голос:").pack(side=tk.LEFT, padx=(0, 10))
        self.speaker_combo = ttk.Combobox(controls_frame, values=SPEAKERS, state="readonly", width=15)
        self.speaker_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.speaker_combo.current(0)  # baya по умолчанию
        logging.debug(f"Комбобокс голосов инициализирован со значениями {SPEAKERS}")
        
        # Настройка скорости озвучки
        ttk.Label(controls_frame, text="Скорость:").pack(side=tk.LEFT, padx=(10, 5))
        self.speech_rate_var = tk.StringVar(value="medium")
        self.speech_rate_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.speech_rate_var,
            values=["x-slow", "slow", "medium", "fast", "x-fast"],
            state="readonly",
            width=10
        )
        self.speech_rate_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.speech_rate_combo.current(2)  # medium по умолчанию
        logging.debug("Комбобокс скорости инициализирован")
        
        # Кнопка воспроизведения
        self.play_btn = ttk.Button(controls_frame, text="▶ Сгенерировать и воспроизвести", command=self.play_audio_threaded)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка сохранения
        self.save_btn = ttk.Button(controls_frame, text="💾 Сохранить (WAV/MP3)", command=self.save_audio_threaded)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка объединения WAV в MP3
        self.merge_mp3_btn = ttk.Button(controls_frame, text="🔗 Объединить в MP3", command=self.merge_wav_to_mp3_threaded)
        self.merge_mp3_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка остановки
        self.stop_btn = ttk.Button(controls_frame, text="⏹ Остановить", command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка открытия папки с аудио
        self.open_folder_btn = ttk.Button(controls_frame, text="📁 Открыть папку аудио", command=self.open_audio_folder)
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)

        # Настройки для больших текстов
        chunk_frame = ttk.Frame(main_frame)
        chunk_frame.pack(fill=tk.X, pady=(5, 0))

        self.chunk_mode_var = tk.BooleanVar(value=False)
        self.save_parts_var = tk.BooleanVar(value=False)
        self.max_chars_var = tk.StringVar(value=str(DEFAULT_MAX_CHARS_PER_CHUNK))
        self.silence_ms_var = tk.StringVar(value=str(DEFAULT_SILENCE_MS))
        
        # Настройки конвертации в MP3
        self.convert_to_mp3_var = tk.BooleanVar(value=False)
        self.mp3_bitrate_var = tk.StringVar(value="192k")
        self.delete_wav_dir_var = tk.BooleanVar(value=False)
        
        # Целевая директория для WAV файлов
        self.target_dir_var = tk.StringVar(value=AUDIO_DIR)

        self.chunk_mode_check = ttk.Checkbutton(
            chunk_frame,
            text="Генерировать частями",
            variable=self.chunk_mode_var,
            command=self.on_chunk_settings_changed
        )
        self.chunk_mode_check.pack(side=tk.LEFT)

        ttk.Label(chunk_frame, text="Макс. символов/часть:").pack(side=tk.LEFT, padx=(10, 4))
        self.max_chars_entry = ttk.Entry(chunk_frame, textvariable=self.max_chars_var, width=6)
        self.max_chars_entry.pack(side=tk.LEFT)
        self.max_chars_entry.bind('<FocusOut>', lambda e: self.on_chunk_settings_changed())
        self.max_chars_entry.bind('<Return>', lambda e: self.on_chunk_settings_changed())

        ttk.Label(chunk_frame, text="Пауза, мс:").pack(side=tk.LEFT, padx=(10, 4))
        self.silence_ms_entry = ttk.Entry(chunk_frame, textvariable=self.silence_ms_var, width=5)
        self.silence_ms_entry.pack(side=tk.LEFT)
        self.silence_ms_entry.bind('<FocusOut>', lambda e: self.on_chunk_settings_changed())
        self.silence_ms_entry.bind('<Return>', lambda e: self.on_chunk_settings_changed())

        self.save_parts_check = ttk.Checkbutton(
            chunk_frame,
            text="Сохранять частями",
            variable=self.save_parts_var,
            command=self.on_chunk_settings_changed
        )
        self.save_parts_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Опция конвертации в MP3
        self.convert_to_mp3_check = ttk.Checkbutton(
            chunk_frame,
            text="Конвертировать в MP3",
            variable=self.convert_to_mp3_var,
            command=self.on_chunk_settings_changed
        )
        self.convert_to_mp3_check.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Label(chunk_frame, text="Битрейт:").pack(side=tk.LEFT, padx=(10, 4))
        self.mp3_bitrate_combo = ttk.Combobox(
            chunk_frame,
            textvariable=self.mp3_bitrate_var,
            values=["128k", "192k", "256k", "320k"],
            width=6,
            state="readonly"
        )
        self.mp3_bitrate_combo.pack(side=tk.LEFT)
        self.mp3_bitrate_combo.bind('<<ComboboxSelected>>', lambda e: self.on_chunk_settings_changed())
        
        # Опция удаления директории WAV после объединения и конвертации
        self.delete_wav_dir_check = ttk.Checkbutton(
            chunk_frame,
            text="Удалять WAV после MP3",
            variable=self.delete_wav_dir_var,
            command=self.on_chunk_settings_changed
        )
        self.delete_wav_dir_check.pack(side=tk.LEFT, padx=(15, 0))
        
        # Целевая директория для WAV файлов
        ttk.Label(chunk_frame, text="Целевая директория:").pack(side=tk.LEFT, padx=(15, 5))
        self.target_dir_entry = ttk.Entry(chunk_frame, textvariable=self.target_dir_var, width=30)
        self.target_dir_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.target_dir_btn = ttk.Button(chunk_frame, text="📁", command=self.select_target_directory)
        self.target_dir_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Фрейм настроек предобработки текста
        preprocess_frame = ttk.Frame(main_frame)
        preprocess_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(preprocess_frame, text="Предобработка текста:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        preprocess_options_frame = ttk.Frame(preprocess_frame)
        preprocess_options_frame.pack(fill=tk.X)
        
        self.use_preprocessing_check = ttk.Checkbutton(
            preprocess_options_frame,
            text="Использовать предобработку текста",
            variable=self.use_preprocessing_var,
            command=self.on_preprocessing_settings_changed
        )
        self.use_preprocessing_check.pack(side=tk.LEFT, padx=(0, 15))
        
        self.use_num2words_check = ttk.Checkbutton(
            preprocess_options_frame,
            text="Заменять числа словами (123 → сто двадцать три)",
            variable=self.use_num2words_var,
            command=self.on_preprocessing_settings_changed
        )
        self.use_num2words_check.pack(side=tk.LEFT, padx=(0, 15))
        
        self.use_ruaccent_check = ttk.Checkbutton(
            preprocess_options_frame,
            text="Доп. обработка (ruaccent, медленнее)",
            variable=self.use_ruaccent_var,
            command=self.on_preprocessing_settings_changed
        )
        self.use_ruaccent_check.pack(side=tk.LEFT)
        
        # Индикатор состояния моделей предобработки
        self.preprocessor_status_var = tk.StringVar(value="Модели предобработки: не загружены")
        preprocessor_status_label = ttk.Label(
            preprocess_options_frame, 
            textvariable=self.preprocessor_status_var, 
            foreground="gray",
            font=("Arial", 8)
        )
        preprocessor_status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Статусная строка (в самом низу под прогресс-баром)
        self.status_var = tk.StringVar(value="Статус: Проверка модели...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue", anchor="w")
        status_label.pack(fill=tk.X, pady=(0, 5))
        
        logging.info("Пользовательский интерфейс настроен успешно")
        
        # Добавление обработчика логов для вывода в текстовое поле
        if hasattr(self, 'log_area'):
            text_handler = TextHandler(self.log_area)
            text_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logging.getLogger().addHandler(text_handler)
            logging.info("Обработчик логов для текстового поля добавлен")
        
        # Применение сохранённых настроек
        self.apply_saved_config()
    
    def apply_saved_config(self):
        """Применение сохранённых настроек"""
        if not self.saved_config:
            logging.debug("Нет сохранённых настроек для применения")
            return
        
        try:
            # Восстановление геометрии окна
            if 'window_geometry' in self.saved_config:
                validated_geometry = self._validate_geometry(self.saved_config['window_geometry'])
                self.root.geometry(validated_geometry)
                logging.debug(f"Геометрия окна восстановлена: {validated_geometry}")
            
            # Восстановление выбора голоса
            if 'speaker' in self.saved_config:
                saved_speaker = self.saved_config['speaker']
                if saved_speaker in SPEAKERS:
                    self.speaker_combo.set(saved_speaker)
                    logging.debug(f"Голос восстановлен: {saved_speaker}")
            
            # Восстановление текста
            if 'text' in self.saved_config and self.saved_config['text']:
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, self.saved_config['text'])
                logging.debug(f"Текст восстановлен (длина: {len(self.saved_config['text'])})")

            if hasattr(self, 'chunk_mode_var'):
                self.chunk_mode_var.set(bool(self.saved_config.get('chunk_mode', False)))
            if hasattr(self, 'save_parts_var'):
                self.save_parts_var.set(bool(self.saved_config.get('save_parts', False)))
            if hasattr(self, 'max_chars_var') and 'max_chars_per_chunk' in self.saved_config:
                self.max_chars_var.set(str(self.saved_config.get('max_chars_per_chunk', DEFAULT_MAX_CHARS_PER_CHUNK)))
            if hasattr(self, 'silence_ms_var') and 'silence_ms' in self.saved_config:
                self.silence_ms_var.set(str(self.saved_config.get('silence_ms', DEFAULT_SILENCE_MS)))
            if hasattr(self, 'chunk_dir_var') and 'chunk_dir' in self.saved_config:
                self.chunk_dir_var.set(self.saved_config.get('chunk_dir', 'my_audiobook'))
            
            # Восстановление настроек MP3
            if hasattr(self, 'convert_to_mp3_var'):
                self.convert_to_mp3_var.set(bool(self.saved_config.get('convert_to_mp3', False)))
            if hasattr(self, 'mp3_bitrate_var') and 'mp3_bitrate' in self.saved_config:
                self.mp3_bitrate_var.set(self.saved_config.get('mp3_bitrate', '192k'))
            if hasattr(self, 'delete_wav_dir_var'):
                self.delete_wav_dir_var.set(bool(self.saved_config.get('delete_wav_dir', False)))
            
            # Восстановление настройки скорости
            if hasattr(self, 'speech_rate_var') and 'speech_rate' in self.saved_config:
                self.speech_rate_var.set(self.saved_config.get('speech_rate', 'medium'))
            
            # Восстановление тестового текста
            if hasattr(self, 'demo_text') and 'demo_text' in self.saved_config:
                self.demo_text = self.saved_config.get('demo_text', DEFAULT_DEMO_TEXT)
            
            # Восстановление целевой директории
            if hasattr(self, 'target_dir_var') and 'target_dir' in self.saved_config:
                self.target_dir_var.set(self.saved_config.get('target_dir', AUDIO_DIR))
            
            # Восстановление настроек предобработки текста
            if hasattr(self, 'use_preprocessing_var'):
                self.use_preprocessing_var.set(bool(self.saved_config.get('use_preprocessing', False)))
            if hasattr(self, 'use_num2words_var'):
                self.use_num2words_var.set(bool(self.saved_config.get('use_num2words', True)))
            if hasattr(self, 'use_ruaccent_var'):
                self.use_ruaccent_var.set(bool(self.saved_config.get('use_ruaccent', False)))
            
            # Восстановление пути к последнему загруженному файлу
            if hasattr(self, 'last_loaded_file_path') and 'last_loaded_file' in self.saved_config:
                self.last_loaded_file_path = self.saved_config.get('last_loaded_file')
                if self.last_loaded_file_path:
                    logging.debug(f"Путь к файлу восстановлен: {self.last_loaded_file_path}")

            self.on_chunk_settings_changed()
            self.on_preprocessing_settings_changed()
        except Exception as e:
            logging.error(f"Ошибка при применении сохранённых настроек: {e}")

    def on_chunk_settings_changed(self):
        try:
            self._get_chunk_settings()
        except Exception:
            return
        try:
            self.save_config()
        except Exception:
            pass
    
    def on_preprocessing_settings_changed(self):
        """Обработка изменений настроек предобработки текста"""
        try:
            use_preprocessing = bool(self.use_preprocessing_var.get()) if hasattr(self, 'use_preprocessing_var') else False
            
            if use_preprocessing and not self.preprocessor_loaded:
                # Загрузка моделей предобработки в отдельном потоке
                self.preprocessor_status_var.set("Модели предобработки: загрузка...")
                thread = threading.Thread(target=self.load_preprocessor_models, daemon=True)
                thread.start()
            elif use_preprocessing and self.preprocessor_loaded:
                self.preprocessor_status_var.set("Модели предобработки: загружены ✓")
            else:
                self.preprocessor_status_var.set("Модели предобработки: не используются")
            
            self.save_config()
        except Exception as e:
            logging.error(f"Ошибка при изменении настроек предобработки: {e}")
    
    def load_preprocessor_models(self):
        """Загрузка моделей предобработки текста в отдельном потоке"""
        try:
            use_ruaccent = bool(self.use_ruaccent_var.get()) if hasattr(self, 'use_ruaccent_var') else False
            logging.info("Начало загрузки моделей предобработки текста")
            self.text_preprocessor.load_models(use_ruaccent=use_ruaccent)
            self.preprocessor_loaded = True
            logging.info("Модели предобработки успешно загружены")
            
            # Обновление статуса в UI
            self.root.after(0, lambda: self.preprocessor_status_var.set("Модели предобработки: загружены ✓"))
            self.root.after(0, lambda: self.update_status("Статус: Модели предобработки загружены ✅"))
            
        except Exception as e:
            logging.error(f"Ошибка при загрузке моделей предобработки: {e}", exc_info=True)
            self.root.after(0, lambda err=e: self.preprocessor_status_var.set("Модели предобработки: ошибка загрузки"))
            self.root.after(0, lambda err=e: self.show_error("Ошибка", f"Не удалось загрузить модели предобработки:\n{err}"))

    def _preprocess_text_for_tts(self, text, status_message="Статус: Предобработка текста...", log_context=""):
        """Применение настроек предобработки текста.

        Важно: замена чисел словами работает независимо от опции
        "Использовать предобработку текста".
        """
        if not text:
            return text

        use_preprocessing = bool(self.use_preprocessing_var.get()) if hasattr(self, 'use_preprocessing_var') else False
        use_num2words = bool(self.use_num2words_var.get()) if hasattr(self, 'use_num2words_var') else True
        use_ruaccent = bool(self.use_ruaccent_var.get()) if hasattr(self, 'use_ruaccent_var') else False

        processed_text = text

        if use_num2words:
            processed_text = self.text_preprocessor.replace_numbers_with_words(processed_text)

        if use_preprocessing and self.preprocessor_loaded:
            self.update_status(status_message)
            processed_text = self.text_preprocessor.preprocess(
                processed_text,
                use_num2words=False,
                use_ruaccent=use_ruaccent
            )
        elif use_preprocessing and not self.preprocessor_loaded:
            logging.warning("Предобработка включена, но модели ещё не загружены. Применена только замена чисел.")

        if processed_text != text:
            context_suffix = f" {log_context}" if log_context else ""
            logging.info(f"Текст предобработан{context_suffix} (длина: {len(processed_text)})")

        return processed_text

    def _get_chunk_settings(self):
        max_chars = int(str(self.max_chars_var.get()).strip())
        if max_chars < 200:
            max_chars = 200
            self.max_chars_var.set(str(max_chars))

        silence_ms = int(str(self.silence_ms_var.get()).strip())
        if silence_ms < 0:
            silence_ms = 0
            self.silence_ms_var.set(str(silence_ms))

        chunk_mode = bool(self.chunk_mode_var.get())
        save_parts = bool(self.save_parts_var.get())
        return chunk_mode, save_parts, max_chars, silence_ms

    def _should_use_chunking(self, text):
        if bool(self.chunk_mode_var.get()):
            return True
        return len(text) >= AUTO_CHUNK_THRESHOLD

    def split_text_into_chunks(self, text, max_chars):
        """
        Разбиение текста на кусочки размером <= max_chars.
        Алгоритм: последовательный проход по тексту слева направо.
        Точка разрыва ищется ТОЛЬКО на границах слов (пробелы), слова не режутся.
        Приоритет: конец предложения (.!?…) → запятая (;:,) → любой пробел.
        """
        text = (text or '').replace('\r\n', '\n').replace('\r', '\n')
        text = ' '.join(text.split())  # Нормализуем пробелы
        if not text:
            return []

        if len(text) <= max_chars:
            return [text]

        chunks = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            remaining = text_len - pos
            if remaining <= max_chars:
                chunk = text[pos:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Окно текста с запасом для поиска точки разрыва
            window_end = min(pos + max_chars, text_len)
            window = text[pos:window_end]

            # Ищем лучшую точку разрыва внутри окна
            break_idx = None

            # 1) Последняя точка конца предложения (.!?…) внутри окна
            for match in re.finditer(r'[.!?…]+', window):
                end_pos = match.end()
                # Проверяем что после знака есть пробел или это конец окна
                if end_pos >= len(window) or window[end_pos] in ' \n':
                    if break_idx is None or end_pos > break_idx:
                        break_idx = end_pos

            # 2) Если не нашли точку предложения, ищем запятую/точку с запятой/двоеточие
            if break_idx is None:
                for match in re.finditer(r'[,;:]', window):
                    end_pos = match.end()
                    if end_pos >= len(window) or window[end_pos] in ' \n':
                        if break_idx is None or end_pos > break_idx:
                            break_idx = end_pos

            # 3) Если не нашли, ищем последний пробел (граница слов)
            if break_idx is None:
                last_space = window.rfind(' ')
                if last_space > 0 and last_space < len(window) - 1:
                    break_idx = last_space + 1

            # 4) Если всё ещё не нашли (очень длинное слово), режем по max_chars
            if break_idx is None:
                break_idx = max_chars

            # Извлекаем чанк и двигаем позицию
            chunk = text[pos:pos + break_idx].strip()
            if chunk:
                chunks.append(chunk)
            pos = pos + break_idx

        # Финальная проверка - добавляем остаток если есть
        if pos < text_len:
            remainder = text[pos:].strip()
            if remainder and (not chunks or remainder != chunks[-1]):
                chunks.append(remainder)

        logging.info(f"Текст разбит на {len(chunks)} чанков (max_chars={max_chars})")
        return chunks

    def _tensor_audio_to_int16_mono(self, audio_tensor):
        audio_numpy = audio_tensor.numpy()
        if audio_numpy.max() > 1.0 or audio_numpy.min() < -1.0:
            max_val = np.abs(audio_numpy).max()
            if max_val > 0:
                audio_numpy = audio_numpy / max_val
        audio_int16 = (audio_numpy * 32767).astype(np.int16)
        if len(audio_int16.shape) == 2 and audio_int16.shape[1] > 1:
            audio_int16 = audio_int16[:, 0]
        elif len(audio_int16.shape) == 2 and audio_int16.shape[1] == 1:
            audio_int16 = audio_int16[:, 0]
        return audio_int16

    def generate_audio_chunked(self, text, speaker, max_chars, silence_ms):
        chunks = self.split_text_into_chunks(text, max_chars)
        if not chunks:
            raise Exception("Пустой текст")

        silence_samples = int(SAMPLE_RATE * (silence_ms / 1000.0))
        silence = np.zeros((silence_samples,), dtype=np.int16) if silence_samples > 0 else None
        
        # Получаем текущую скорость озвучки
        speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'

        # Запускаем прогресс-бар с количеством чанков
        self.start_progress(total_chunks=len(chunks))

        audio_parts = []
        for idx, chunk_text in enumerate(chunks, start=1):
            # Проверка флага остановки
            self.check_stop_flag()
            
            self.update_status(f"Статус: Генерация части {idx}/{len(chunks)}...")
            audio_tensor = self.generate_audio(chunk_text, speaker, speech_rate)
            audio_int16 = self._tensor_audio_to_int16_mono(audio_tensor)
            audio_parts.append(audio_int16)
            if silence is not None and idx != len(chunks):
                audio_parts.append(silence)
            
            # Обновляем прогресс-бар
            self.update_progress(idx)

        return np.concatenate(audio_parts), chunks

    def split_text_to_chunks_ui(self):
        try:
            text = self.text_area.get("1.0", tk.END).strip()
            _, save_parts, max_chars, _silence_ms = self._get_chunk_settings()

            if not text:
                self.show_warning("Внимание", "Введите текст для разделения")
                return

            text_for_display = self._preprocess_text_for_tts(
                text,
                status_message="Статус: Предобработка текста для чанков...",
                log_context="для чанков"
            )
            
            chunks = self.split_text_into_chunks(text_for_display, max_chars)
            if not chunks:
                self.show_warning("Внимание", "Не удалось получить кусочки")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_dir = self.chunk_dir_var.get().strip() if hasattr(self, 'chunk_dir_var') else "my_audiobook"
            if not custom_dir:
                custom_dir = "my_audiobook"
            base_name = f"{custom_dir}_{timestamp}"
            parts_dir_name = f"{base_name}_parts"

            self.last_chunk_plan = {
                'timestamp': timestamp,
                'speaker': self.speaker_combo.get(),
                'base_name': base_name,
                'parts_dir': os.path.join(AUDIO_DIR, parts_dir_name),
                'save_parts': save_parts,
                'max_chars': max_chars,
                'chunks': chunks,
                'original_text': text,
                'preprocessed_text': text_for_display
            }

            # Получаем текущую скорость озвучки
            speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'
            
            self.chunks_area.delete("1.0", tk.END)
            self.chunks_area.insert(tk.END, f"План чанков: {base_name}\n")
            self.chunks_area.insert(tk.END, f"Папка частей: {parts_dir_name}\\part_###.wav\n")
            self.chunks_area.insert(tk.END, f"Чанков: {len(chunks)}\n")
            self.chunks_area.insert(tk.END, f"Скорость: {speech_rate}\n\n")

            for idx, chunk_text in enumerate(chunks, start=1):
                # Логирование для отладки: содержимое chunk_text до обработки
                logging.debug(f"CHUNK {idx:03d} (до обработки): '{chunk_text[:100]}...' (длина: {len(chunk_text)})")
                
                # Экранируем XML-символы в chunk_text перед вставкой в SSML
                # Это сохраняет ударения и другие специальные символы
                chunk_text_escaped = html.escape(chunk_text, quote=False)
                
                # Добавляем теги пауз после точек в конце предложений
                chunk_with_pauses = self.add_pause_tags(chunk_text_escaped)
                
                # Оборачиваем каждый чанк в SSML теги скорости
                chunk_with_ssml = f'<speak><prosody rate="{speech_rate}">{chunk_with_pauses}</prosody></speak>'
                
                # Логирование для отладки: финальный SSML
                logging.debug(f"CHUNK {idx:03d} (SSML): '{chunk_with_ssml[:120]}...' (длина: {len(chunk_with_ssml)})")
                
                self.chunks_area.insert(
                    tk.END,
                    f"==== CHUNK {idx:03d} -> part_{idx:03d}.wav ====\n{chunk_with_ssml}\n\n"
                )

            self.update_status(f"Статус: Текст разделён на кусочки ({len(chunks)}) ✅")
            try:
                self.notebook.select(1)
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Ошибка при разделении на кусочки: {e}", exc_info=True)
            self.show_error("Ошибка", str(e))

    def _parse_chunks_from_ui(self):
        """Парсинг чанков из chunks_area (извлекает текст между ==== CHUNK ... ==== и следующим ====)"""
        try:
            chunks_content = self.chunks_area.get("1.0", tk.END).strip()
            if not chunks_content:
                return []
            
            # Разделяем по маркерам CHUNK - более надёжный regex
            # Ищем все вхождения ==== CHUNK X -> part_X.wav ====
            chunk_markers = list(re.finditer(r'==== CHUNK (\d+) -> part_(\d+)\.wav ====', chunks_content))
            
            if not chunk_markers:
                logging.warning("Не найдено маркеров CHUNK в chunks_area")
                return []
            
            parsed_chunks = []
            for i, match in enumerate(chunk_markers):
                chunk_num = int(match.group(1))
                start_pos = match.end()
                
                # Конец чанка - начало следующего маркера или конец текста
                if i + 1 < len(chunk_markers):
                    end_pos = chunk_markers[i + 1].start()
                else:
                    end_pos = len(chunks_content)
                
                # Извлекаем текст чанка
                chunk_text = chunks_content[start_pos:end_pos].strip()
                
                if chunk_text:
                    # Удаляем переносы строк из текста чанка
                    chunk_text = chunk_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                    # Удаляем символы = и пробелы в конце (артефакты UI)
                    chunk_text = chunk_text.rstrip(' =\t')
                    # Удаляем множественные пробелы
                    chunk_text = ' '.join(chunk_text.split())
                    
                    parsed_chunks.append((chunk_num, chunk_text))
            
            # Сортируем чанки по номеру и возвращаем только текст
            parsed_chunks.sort(key=lambda x: x[0])
            result = [text for _, text in parsed_chunks]
            
            logging.info(f"Распарсено {len(result)} чанков из UI (найдено маркеров: {len(chunk_markers)})")
            return result
        except Exception as e:
            logging.error(f"Ошибка при парсинге чанков из UI: {e}", exc_info=True)
            return []

    def speak_chunks_threaded(self):
        if not self.is_model_loaded:
            self.show_warning("Внимание", "Модель ещё загружается. Пожалуйста, подождите.")
            return
        thread = threading.Thread(target=self.speak_chunks, daemon=True)
        thread.start()

    def speak_chunks(self):
        try:
            chunks = None
            chunks_source = None
            
            # 1. Сначала пытаемся получить чанки из UI (chunks_area) - там уже полные SSML-теги
            if hasattr(self, 'chunks_area'):
                chunks = self._parse_chunks_from_ui()
                if chunks:
                    chunks_source = "UI (chunks_area с SSML)"
                    logging.info(f"Используются чанки из UI (количество: {len(chunks)})")
            
            # 2. Если в UI нет чанков, пробуем last_chunk_plan
            if not chunks and hasattr(self, 'last_chunk_plan') and self.last_chunk_plan and 'chunks' in self.last_chunk_plan:
                chunks = self.last_chunk_plan['chunks']
                chunks_source = "last_chunk_plan"
                logging.info(f"Используются готовые чанки из last_chunk_plan (количество: {len(chunks)})")
            
            # 3. Если чанков нет, разбиваем текст заново
            if not chunks:
                text = self.text_area.get("1.0", tk.END).strip()
                if not text:
                    self.show_warning("Внимание", "Введите текст для озвучки")
                    return

                text = self._preprocess_text_for_tts(text)
                
                _, _, max_chars, _ = self._get_chunk_settings()
                chunks = self.split_text_into_chunks(text, max_chars)
                chunks_source = "разбиение текста"
            
            if not chunks:
                self.show_warning("Внимание", "Не удалось получить кусочки")
                return

            speaker = self.speaker_combo.get()
            chunk_mode, save_parts, max_chars, silence_ms = self._get_chunk_settings()
            
            # Сброс флага остановки перед началом генерации
            self.reset_stop_flag()
            
            self.update_status("Статус: Озвучивание чанков...")
            self.start_progress()

            # Получение целевой директории
            target_dir = self.target_dir_var.get() if hasattr(self, 'target_dir_var') else AUDIO_DIR
            
            # Создание директории для аудио, если не существует
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                logging.info(f"Создана директория для аудио: {target_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_dir = self.chunk_dir_var.get().strip() if hasattr(self, 'chunk_dir_var') else "my_audiobook"
            if not custom_dir:
                custom_dir = "my_audiobook"
            base_name = f"{custom_dir}_{timestamp}"
            full_path = os.path.join(target_dir, f"{base_name}.wav")
            parts_dir = os.path.join(target_dir, f"{base_name}_parts")
            if save_parts:
                os.makedirs(parts_dir, exist_ok=True)

            silence_samples = int(SAMPLE_RATE * (silence_ms / 1000.0))
            silence = np.zeros((silence_samples,), dtype=np.int16) if silence_samples > 0 else None
            
            # Получаем текущую скорость озвучки
            speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'

            # Запускаем прогресс-бар с количеством чанков
            self.start_progress(total_chunks=len(chunks))

            audio_parts = []
            for idx, chunk_text in enumerate(chunks, start=1):
                # Проверка флага остановки
                self.check_stop_flag()
                
                self.update_status(f"Статус: Генерация части {idx}/{len(chunks)}...")
                
                # При использовании чанков из UI, передаём в generate_audio() уже готовый SSML-текст
                # generate_audio() сам распознает SSML теги и использует их без дополнительной обёртки
                audio_tensor = self.generate_audio(chunk_text, speaker, speech_rate)
                audio_int16 = self._tensor_audio_to_int16_mono(audio_tensor)

                if save_parts:
                    part_path = os.path.join(parts_dir, f"part_{idx:03d}.wav")
                    self._write_wav_int16_mono(part_path, audio_int16)

                audio_parts.append(audio_int16)
                if silence is not None and idx != len(chunks):
                    audio_parts.append(silence)
                
                # Обновляем прогресс-бар
                self.update_progress(idx)

            audio_full_int16 = np.concatenate(audio_parts) if audio_parts else np.zeros((0,), dtype=np.int16)
            self._write_wav_int16_mono(full_path, audio_full_int16)

            # Конвертация в MP3 если включена опция
            convert_to_mp3 = bool(self.convert_to_mp3_var.get()) if hasattr(self, 'convert_to_mp3_var') else False
            mp3_path = None
            if convert_to_mp3:
                self.update_status(f"Статус: Конвертация в MP3...")
                bitrate = self.mp3_bitrate_var.get() if hasattr(self, 'mp3_bitrate_var') else "192k"
                try:
                    mp3_path = self.convert_wav_to_mp3(full_path, bitrate)
                    logging.info(f"MP3 файл создан: {mp3_path}")
                    
                    # Удаляем WAV файл после успешной конвертации в MP3
                    try:
                        os.remove(full_path)
                        logging.info(f"WAV файл удалён: {full_path}")
                    except Exception as remove_error:
                        logging.warning(f"Не удалось удалить WAV файл: {remove_error}")
                    
                    # Удаляем директорию с частями если включена опция
                    delete_wav_dir = bool(self.delete_wav_dir_var.get()) if hasattr(self, 'delete_wav_dir_var') else False
                    if delete_wav_dir and save_parts:
                        logging.info(f"=== УДАЛЕНИЕ ДИРЕКТОРИИ С ЧАСТЯМИ (speak_chunks) ===")
                        logging.info(f"  delete_wav_dir: {delete_wav_dir}")
                        logging.info(f"  save_parts: {save_parts}")
                        logging.info(f"  parts_dir: {parts_dir}")
                        logging.info(f"  parts_dir существует: {os.path.exists(parts_dir)}")
                        try:
                            import shutil
                            if os.path.exists(parts_dir):
                                shutil.rmtree(parts_dir)
                                logging.info(f"Директория с частями удалена: {parts_dir}")
                                logging.info(f"parts_dir существует после удаления: {os.path.exists(parts_dir)}")
                            else:
                                logging.warning(f"Директория с частями не найдена: {parts_dir}")
                        except Exception as del_error:
                            logging.error(f"Не удалось удалить директорию с частями: {del_error}", exc_info=True)
                except Exception as mp3_error:
                    logging.error(f"Ошибка конвертации в MP3: {mp3_error}")
                    self.show_warning("Предупреждение", f"WAV сохранён, но конвертация в MP3 не удалась:\n{str(mp3_error)}")
            
            self.update_status(f"Статус: Сохранено (кусочки) ✅")
            if mp3_path:
                mp3_size = os.path.getsize(mp3_path) / 1024
                extra = f"\n\nЧасти: {parts_dir}" if save_parts else ""
                message = f"Аудио сохранено в MP3 файл:\n{mp3_path}\n\nРазмер: {mp3_size:.2f} КБ, битрейт: {bitrate}{extra}"
            else:
                wav_size = os.path.getsize(full_path) / 1024
                extra = f"\n\nЧасти: {parts_dir}" if save_parts else ""
                message = f"Аудио сохранено в файл:\n{full_path}\n\nРазмер: {wav_size:.2f} КБ{extra}"
            self.show_info("Успех", message)
            self.stop_progress()
            
        except InterruptedError:
            # Обработка остановки пользователем
            logging.info("Озвучивание чанков остановлено пользователем")
            self.update_status("Статус: Озвучивание остановлено пользователем")
            self.stop_progress()
            
        except Exception as e:
            logging.error(f"Ошибка при озвучивании чанков: {e}", exc_info=True)
            self.update_status("Статус: Ошибка ❌")
            self.show_error("Ошибка", str(e))
            self.stop_progress()

    def _extract_fb2_from_zip(self, zip_path):
        """Извлечение FB2 файла из ZIP архива и парсинг текста"""
        import zipfile
        import xml.etree.ElementTree as ET
        import io
        
        logging.info(f"Распаковка ZIP архива: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Поиск FB2 файлов в архиве
            fb2_files = [f for f in zf.namelist() if f.lower().endswith('.fb2')]
            
            if not fb2_files:
                raise Exception("В ZIP архиве не найдено файлов формата FB2")
            
            if len(fb2_files) > 1:
                logging.warning(f"Найдено несколько FB2 файлов: {fb2_files}. Будет использован первый: {fb2_files[0]}")
            
            fb2_filename = fb2_files[0]
            logging.info(f"Извлечение FB2 файла: {fb2_filename}")
            
            # Чтение содержимого FB2 файла
            with zf.open(fb2_filename) as fb2_file:
                fb2_content = fb2_file.read()
                
            # Парсинг FB2 из памяти
            root = ET.parse(io.BytesIO(fb2_content)).getroot()
            
            # Извлечение текста из FB2
            text_parts = []
            for elem in root.iter():
                if elem.tag.endswith('p') or elem.tag.endswith('title'):
                    if elem.text:
                        text_parts.append(elem.text.strip())
            
            text = '\n'.join(text_parts)
            logging.info(f"Текст извлечён из FB2 в ZIP (длина: {len(text)})")
            
            return text
    
    def _write_wav_int16_mono(self, path, audio_int16):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
    
    def load_file(self):
        """Загрузка текста из файла txt, fb2 или zip с fb2 внутри"""
        try:
            from tkinter import filedialog
            import zipfile
            
            filetypes = [
                ('Текстовые файлы', '*.txt;*.md'),
                ('FB2 файлы', '*.fb2'),
                ('ZIP архивы с FB2', '*.zip'),
                ('Все файлы', '*.*')
            ]
            
            file_path = filedialog.askopenfilename(
                title="Выберите файл для озвучки",
                filetypes=filetypes
            )
            
            if not file_path:
                return
            
            logging.info(f"Выбран файл: {file_path}")
            
            # Определение формата файла
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.zip':
                # Поиск FB2 файла внутри ZIP архива
                text = self._extract_fb2_from_zip(file_path)
            elif file_ext == '.fb2':
                # Парсинг FB2 (простой вариант - извлечение текста из тегов <p>)
                import xml.etree.ElementTree as ET
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Извлечение текста из FB2
                text_parts = []
                for elem in root.iter():
                    if elem.tag.endswith('p') or elem.tag.endswith('title'):
                        if elem.text:
                            text_parts.append(elem.text.strip())
                
                text = '\n'.join(text_parts)
                logging.info(f"Текст извлечён из FB2 (длина: {len(text)})")
                
            else:
                # Чтение TXT файла
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                logging.info(f"Текст загружен из TXT (длина: {len(text)})")
            
            # Вставка текста в текстовое поле
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, text)

            try:
                if hasattr(self, 'chunks_area'):
                    self.chunks_area.delete("1.0", tk.END)
            except Exception:
                pass
            
            # Сохранение пути к файлу для использования в CLI команде
            self.last_loaded_file_path = file_path
            self.save_config()
            
            self.update_status(f"Статус: Файл загружён: {os.path.basename(file_path)} ✅")
            logging.info(f"Текст из файла {file_path} успешно загружен в текстовое поле")
            
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла: {e}", exc_info=True)
            self.show_error("Ошибка загрузки файла", f"Не удалось загрузить файл: {e}")
    
    def load_demo_text(self):
        """Загрузка тестового текста из настроек"""
        try:
            demo_text = self.demo_text if hasattr(self, 'demo_text') else DEFAULT_DEMO_TEXT
            
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, demo_text)
            
            try:
                if hasattr(self, 'chunks_area'):
                    self.chunks_area.delete("1.0", tk.END)
            except Exception:
                pass
            
            self.update_status(f"Статус: Тестовый текст загружён ✅")
            logging.info(f"Тестовый текст загружён (длина: {len(demo_text)})")
            
        except Exception as e:
            logging.error(f"Ошибка при загрузке тестового текста: {e}", exc_info=True)
            self.show_error("Ошибка", f"Не удалось загрузить тестовый текст: {e}")
    
    def generate_cli_command(self):
        """Генерация CLI команды на основе текущих настроек"""
        try:
            # Получение текущих настроек
            speaker = self.speaker_combo.get() if hasattr(self, 'speaker_combo') else 'baya'
            speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'
            chunk_mode = bool(self.chunk_mode_var.get()) if hasattr(self, 'chunk_mode_var') else False
            save_parts = bool(self.save_parts_var.get()) if hasattr(self, 'save_parts_var') else False
            max_chars = int(str(self.max_chars_var.get()).strip()) if hasattr(self, 'max_chars_var') else DEFAULT_MAX_CHARS_PER_CHUNK
            silence_ms = int(str(self.silence_ms_var.get()).strip()) if hasattr(self, 'silence_ms_var') else DEFAULT_SILENCE_MS
            convert_to_mp3 = bool(self.convert_to_mp3_var.get()) if hasattr(self, 'convert_to_mp3_var') else False
            mp3_bitrate = self.mp3_bitrate_var.get() if hasattr(self, 'mp3_bitrate_var') else '192k'
            delete_wav_dir = bool(self.delete_wav_dir_var.get()) if hasattr(self, 'delete_wav_dir_var') else False
            use_preprocessing = bool(self.use_preprocessing_var.get()) if hasattr(self, 'use_preprocessing_var') else False
            use_num2words = bool(self.use_num2words_var.get()) if hasattr(self, 'use_num2words_var') else True
            use_ruaccent = bool(self.use_ruaccent_var.get()) if hasattr(self, 'use_ruaccent_var') else False
            target_dir = self.target_dir_var.get() if hasattr(self, 'target_dir_var') else AUDIO_DIR
            
            # Формирование базовой команды
            cmd_parts = ['python text2mp3.py']
            
            # Текст или файл
            text = self.text_area.get("1.0", tk.END).strip()
            
            # Проверяем, есть ли сохранённый путь к файлу (из JSON или текущей сессии)
            file_path = None
            if hasattr(self, 'last_loaded_file_path') and self.last_loaded_file_path:
                file_path = self.last_loaded_file_path
            elif hasattr(self, 'saved_config') and self.saved_config.get('last_loaded_file'):
                file_path = self.saved_config.get('last_loaded_file')
            
            if file_path:
                cmd_parts.append(f'--input-file "{file_path}"')
            elif text:
                # Если текст короткий, добавляем его в команду
                if len(text) < 200:
                    escaped_text = text.replace('"', '\\"')
                    cmd_parts.append(f'--text "{escaped_text}"')
                else:
                    cmd_parts.append('--input-file <путь_к_файлу>')
            
            # Выходной файл
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_ext = '.mp3' if convert_to_mp3 else '.wav'
            cmd_parts.append(f'--output output_{timestamp}{output_ext}')
            
            # Голос и скорость
            if speaker != 'baya':
                cmd_parts.append(f'--speaker {speaker}')
            if speech_rate != 'medium':
                cmd_parts.append(f'--speech-rate {speech_rate}')
            
            # Чанки
            if chunk_mode or (text and len(text) > AUTO_CHUNK_THRESHOLD):
                cmd_parts.append('--chunks')
                if max_chars != DEFAULT_MAX_CHARS_PER_CHUNK:
                    cmd_parts.append(f'--max-chars {max_chars}')
                if silence_ms != DEFAULT_SILENCE_MS:
                    cmd_parts.append(f'--silence-ms {silence_ms}')
                if save_parts:
                    cmd_parts.append('--save-parts')
            
            # MP3 конвертация
            if convert_to_mp3:
                cmd_parts.append('--mp3')
                if mp3_bitrate != '192k':
                    cmd_parts.append(f'--bitrate {mp3_bitrate}')
                if delete_wav_dir:
                    cmd_parts.append('--delete-parts')
            
            # Предобработка
            if use_preprocessing:
                cmd_parts.append('--preprocess')
                if not use_num2words:
                    cmd_parts.append('--no-num2words')
                if use_ruaccent:
                    cmd_parts.append('--ruaccent')
            
            # Целевая директория
            if target_dir != AUDIO_DIR:
                cmd_parts.append(f'--output-dir "{target_dir}"')
            
            # Сборка команды
            cli_command = ' '.join(cmd_parts)
            
            # Показ команды
            self.show_info("CLI команда", f"Команда для запуска из консоли:\n\n{cli_command}\n\nНажмите Ctrl+C чтобы скопировать")
            
            # Копирование в буфер обмена
            self.root.clipboard_clear()
            self.root.clipboard_append(cli_command)
            self.update_status("Статус: CLI команда скопирована в буфер обмена ✅")
            logging.info(f"CLI команда сгенерирована: \n\n{cli_command}\n\n")
            
        except Exception as e:
            logging.error(f"Ошибка при генерации CLI команды: {e}", exc_info=True)
            self.show_error("Ошибка", f"Не удалось создать CLI команду:\n{e}")
    
    def select_target_directory(self):
        """Выбор целевой директории для WAV файлов"""
        try:
            from tkinter import filedialog
            
            # Выбор директории
            directory = filedialog.askdirectory(
                title="Выберите директорию для сохранения WAV файлов",
                initialdir=self.target_dir_var.get() if hasattr(self, 'target_dir_var') else AUDIO_DIR
            )
            
            if directory:
                self.target_dir_var.set(directory)
                logging.info(f"Целевая директория установлена: {directory}")
                self.save_config()
                self.update_status(f"Статус: Целевая директория: {directory} ✅")
            
        except Exception as e:
            logging.error(f"Ошибка при выборе директории: {e}", exc_info=True)
            self.show_error("Ошибка", f"Не удалось выбрать директорию: {e}")
    
    def merge_wav_to_mp3_threaded(self):
        """Запуск объединения WAV в MP3 в отдельном потоке"""
        try:
            from tkinter import filedialog
            
            # Получение целевой директории по умолчанию
            default_dir = AUDIO_DIR
            if hasattr(self, 'target_dir_var'):
                default_dir = self.target_dir_var.get()
            elif hasattr(self, 'saved_config') and 'target_dir' in self.saved_config:
                default_dir = self.saved_config['target_dir']
            
            # Выбор директории с WAV файлами (по умолчанию - целевая директория)
            directory = filedialog.askdirectory(
                title="Выберите директорию с WAV файлами",
                initialdir=default_dir
            )
            if not directory:
                return
            
            logging.info(f"Выбрана директория: {directory}")
            
            # Запуск в отдельном потоке
            thread = threading.Thread(target=self.merge_wav_to_mp3, args=(directory,), daemon=True)
            thread.start()
            
        except Exception as e:
            logging.error(f"Ошибка при выборе директории: {e}", exc_info=True)
            self.show_error("Ошибка", f"Не удалось выбрать директорию: {e}")
    
    def merge_wav_to_mp3(self, directory):
        """Объединение WAV файлов из директории в один MP3 файл"""
        try:
            self.reset_stop_flag()
            
            logging.info(f"=== Начало объединения WAV в MP3 из: {directory} ===")
            self.update_status("Статус: Поиск WAV файлов...")
            
            # Поиск всех WAV файлов в директории
            wav_files = sorted([
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith('.wav') and not f.startswith('~')
            ])
            
            if not wav_files:
                self.show_warning("Внимание", f"В директории не найдено WAV файлов:\n{directory}")
                self.update_status("Статус: WAV файлы не найдены")
                return
            
            logging.info(f"Найдено WAV файлов: {len(wav_files)}")
            self.update_status(f"Статус: Найдено {len(wav_files)} WAV файлов")
            
            # Запуск прогресс-бара
            self.start_progress(total_chunks=len(wav_files))
            
            # Загрузка и объединение аудио
            audio_parts = []
            for idx, wav_path in enumerate(wav_files, start=1):
                # Проверка флага остановки
                self.check_stop_flag()
                
                self.update_status(f"Статус: Обработка {idx}/{len(wav_files)}...")
                logging.info(f"Загрузка WAV: {wav_path} ({idx}/{len(wav_files)})")
                
                try:
                    audio = AudioSegment.from_wav(wav_path)
                    audio_parts.append(audio)
                except Exception as e:
                    logging.warning(f"Не удалось загрузить {wav_path}: {e}")
                
                # Обновление прогресс-бара
                self.update_progress(idx)
            
            if not audio_parts:
                self.show_error("Ошибка", "Не удалось загрузить ни один WAV файл")
                self.update_status("Статус: Ошибка объединения")
                self.stop_progress()
                return
            
            # Объединение аудио
            self.update_status("Статус: Объединение аудио...")
            logging.info("Объединение аудио файлов")
            combined_audio = audio_parts[0]
            for audio in audio_parts[1:]:
                combined_audio += audio
            
            # Получение битрейта из настроек
            bitrate = self.mp3_bitrate_var.get() if hasattr(self, 'mp3_bitrate_var') else "192k"
            
            # Создание MP3 файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mp3_path = os.path.join(directory, f"combined_{timestamp}.mp3")
            
            self.update_status(f"Статус: Конвертация в MP3 (битрейт: {bitrate})...")
            logging.info(f"Экспорт в MP3: {mp3_path} (битрейт: {bitrate})")
            
            combined_audio.export(mp3_path, format="mp3", bitrate=bitrate)
            
            if os.path.exists(mp3_path):
                mp3_size = os.path.getsize(mp3_path) / 1024
                # Расчёт длительности аудио в секундах
                duration_seconds = len(combined_audio) / 1000.0  # pydub использует миллисекунды
                duration_hours = int(duration_seconds // 3600)
                duration_minutes = int((duration_seconds % 3600) // 60)
                duration_secs_remaining = int(duration_seconds % 60)
                duration_formatted = f"{duration_hours}:{duration_minutes:02d}:{duration_secs_remaining:02d}"
                
                logging.info(f"MP3 файл создан: {mp3_path} (размер: {mp3_size:.2f} КБ, битрейт: {bitrate}, длительность: {duration_formatted})")
                
                # Удаление директории с WAV файлами если включена опция
                delete_wav_dir = bool(self.delete_wav_dir_var.get()) if hasattr(self, 'delete_wav_dir_var') else False
                logging.info(f"=== ПРОВЕРКА ОПЦИИ УДАЛЕНИЯ WAV ===")
                logging.info(f"  delete_wav_dir_var существует: {hasattr(self, 'delete_wav_dir_var')}")
                logging.info(f"  Значение delete_wav_dir: {delete_wav_dir}")
                logging.info(f"  Количество WAV файлов: {len(wav_files)}")
                logging.info(f"  Директория: {directory}")
                logging.info(f"  Директория существует: {os.path.exists(directory)}")
                
                if delete_wav_dir:
                    logging.info(f"Опция удаления ВКЛЮЧЕНА, начинаем удаление...")
                    if wav_files:
                        try:
                            import shutil
                            logging.info(f"Удаление директории с WAV файлами: {directory}")
                            logging.info(f"Содержимое директории перед удалением: {os.listdir(directory)}")
                            shutil.rmtree(directory)
                            logging.info(f"Директория успешно удалена: {directory}")
                            logging.info(f"Директория существует после удаления: {os.path.exists(directory)}")
                            self.update_status(f"Статус: WAV файлы удалены ✅")
                        except Exception as del_error:
                            logging.error(f"КРИТИЧЕСКАЯ ОШИБКА при удалении директории: {del_error}", exc_info=True)
                            logging.error(f"Тип ошибки: {type(del_error).__name__}")
                            self.show_warning("Предупреждение", f"MP3 создан, но не удалось удалить WAV файлы:\n{del_error}")
                    else:
                        logging.warning("Опция удаления включена, но wav_files пуст")
                else:
                    logging.info("Опция удаления WAV ВЫКЛЮЧЕНА")
                
                self.update_status(f"Статус: MP3 создан ✅")
                self.show_info(
                    "Успех", 
                    f"Создан MP3 файл:\n{mp3_path}\n\n"
                    f"Объединено файлов: {len(audio_parts)}\n"
                    f"Размер: {mp3_size:.2f} КБ\n"
                    f"Битрейт: {bitrate}\n"
                    f"Длительность: {duration_formatted} ({duration_seconds:.1f} сек)"
                )
            else:
                raise Exception("MP3 файл не был создан")
            
            self.stop_progress()
            
        except InterruptedError:
            logging.info("Объединение остановлено пользователем")
            self.update_status("Статус: Объединение остановлено")
            self.stop_progress()
            
        except Exception as e:
            logging.error(f"Ошибка при объединении WAV в MP3: {e}", exc_info=True)
            self.update_status("Статус: Ошибка объединения ❌")
            self.show_error("Ошибка", f"Не удалось объединить WAV файлы:\n{e}")
            self.stop_progress()
    
    def load_model_threaded(self):
        """Загрузка модели в отдельном потоке"""
        logging.info("Запуск потока для загрузки модели")
        thread = threading.Thread(target=self.load_model, daemon=True)
        thread.start()
    
    def load_model(self):
        """Загрузка модели Silero TTS v5"""
        try:
            logging.info("=== Начало процесса загрузки модели ===")
            self.update_status("Статус: Загрузка модели...")
            self.start_progress()
            
            # Создание директории кэша, если не существует
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR, exist_ok=True)
                logging.info(f"Создана директория кэша: {CACHE_DIR}")
            
            # Проверка наличия файла модели
            if not os.path.isfile(MODEL_FILE):
                logging.info(f"Файл модели {MODEL_FILE} не найден. Начинается скачивание...")
                self.update_status("Статус: Скачивание модели в кэш (первый запуск может занять время)...")
                
                # Проверка доступности URL
                try:
                    torch.hub.download_url_to_file(MODEL_URL, MODEL_FILE, progress=True)
                    file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
                    logging.info(f"Модель успешно скачана: {MODEL_FILE} ({file_size:.2f} МБ)")
                except Exception as e:
                    logging.error(f"Ошибка при скачивании модели: {e}", exc_info=True)
                    raise Exception(f"Не удалось скачать модель: {e}")
            else:
                file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)  # в МБ
                logging.info(f"Файл модели {MODEL_FILE} уже существует в кэше (размер: {file_size:.2f} МБ)")
            
            # Проверка целостности файла
            if os.path.getsize(MODEL_FILE) < 1000000:  # Меньше 1 МБ
                logging.warning(f"Файл модели подозрительно мал: {os.path.getsize(MODEL_FILE)} байт")
            
            # Загрузка модели
            logging.info("Начинается загрузка модели в память...")
            device = torch.device('cpu')
            torch.set_num_threads(4)
            logging.debug(f"Установлено количество потоков CPU: 4, устройство: {device}")
            
            try:
                self.model = torch.package.PackageImporter(MODEL_FILE).load_pickle("tts_models", "model")
                self.model.to(device)
                logging.info("Модель успешно загружена в память")
                
                # Проверка модели
                if hasattr(self.model, 'speakers'):
                    logging.info(f"Доступные спикеры в модели: {self.model.speakers}")
                else:
                    logging.warning("Модель не имеет атрибута speakers")
                
            except Exception as e:
                logging.error(f"Ошибка при загрузке модели в память: {e}", exc_info=True)
                raise Exception(f"Не удалось загрузить модель в память: {e}")
            
            self.is_model_loaded = True
            self.update_status("Статус: Модель загружена ✅ Готов к работе")
            self.stop_progress()
            logging.info("=== Процесс загрузки модели завершен успешно ===")
            
        except Exception as e:
            logging.error(f"Критическая ошибка при загрузке модели: {str(e)}", exc_info=True)
            self.update_status("Статус: Ошибка загрузки ❌")
            self.show_error("Ошибка загрузки модели", str(e))
            self.stop_progress()
    
    def generate_audio(self, text, speaker, speech_rate=None):
        """Генерация аудио из текста с поддержкой скорости через SSML"""
        if not self.is_model_loaded:
            raise Exception("Модель не загружена")
        
        if speech_rate is None:
            speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'
        
        try:
            # Очистка текста от неподдерживаемых XML-символов
            # Silero TTS v5 не поддерживает переносы строк, русские кавычки и другие спецсимволы в SSML
            def clean_xml_text(text):
                # Удаляем переносы строк (заменяем на пробелы)
                text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                # Заменяем русские кавычки на стандартные
                text = text.replace('«', '"').replace('»', '"')
                # Заменяем тире на дефис
                text = text.replace('—', '-').replace('–', '-')
                # Заменяем многоточие на три точки
                text = text.replace('…', '...')
                # Экранируем амперсанд
                text = text.replace('&', '&amp;')
                # Буква ё поддерживается - оставляем как есть
                # Удаляем множественные пробелы
                text = ' '.join(text.split())
                return text
            
            # Проверяем, содержит ли текст уже SSML теги <speak>
            text_stripped = text.strip()
            if text_stripped.startswith('<speak>') and '</speak>' in text_stripped:
                # Текст уже содержит SSML, используем как есть
                # Но всё равно очищаем содержимое от неподдерживаемых символов
                import re
                # Извлекаем содержимое между <prosody ...> и </prosody>
                match = re.search(r'<prosody[^>]*>(.*?)</prosody>', text_stripped, re.DOTALL)
                if match:
                    inner_text = match.group(1)
                    cleaned_inner = clean_xml_text(inner_text)
                    ssml_text = text_stripped.replace(inner_text, cleaned_inner)
                    logging.info("Текст уже содержит SSML теги, очищаем от неподдерживаемых символов")
                else:
                    ssml_text = text_stripped
            else:
                # Обёртка текста в SSML для управления скоростью
                cleaned_text = clean_xml_text(text)
                # Добавляем теги пауз после точек в конце предложений
                text_with_pauses = self.add_pause_tags(cleaned_text)
                ssml_text = f'<speak><prosody rate="{speech_rate}">{text_with_pauses}</prosody></speak>'
            
            # Логирование финального SSML-текста с тегами скорости и ударениями
            logging.info(f"Генерация аудио: текст='{ssml_text[:100]}...' (длина: {len(ssml_text)}), голос='{speaker}', скорость='{speech_rate}')")
            
            # Генерация аудио с SSML
            audio = self.model.apply_tts(
                ssml_text=ssml_text,
                speaker=speaker,
                sample_rate=SAMPLE_RATE,
                put_accent=True,
                put_yo=True
            )
            
            logging.info(f"Аудио сгенерировано успешно. Тип: {type(audio)}, форма: {audio.shape if hasattr(audio, 'shape') else 'неизвестно'}")
            return audio
            
        except Exception as e:
            logging.error(f"Ошибка при генерации аудио: {e}", exc_info=True)
            raise
    
    def update_status(self, message):
        """Безопасное обновление статуса в UI"""
        try:
            self.root.after(0, lambda: self.status_var.set(message))
            logging.debug(f"Статус обновлен: {message}")
        except Exception as e:
            logging.error(f"Ошибка при обновлении статуса: {e}")
    
    def start_progress(self, total_chunks=None):
        """Запуск прогресс-бара"""
        try:
            if total_chunks is not None and total_chunks > 0:
                self.progress['maximum'] = total_chunks
                self.progress['value'] = 0
            else:
                self.progress['maximum'] = 100
                self.progress['value'] = 0
        except Exception as e:
            logging.error(f"Ошибка при запуске прогресс-бара: {e}")
    
    def update_progress(self, value=None):
        """Обновление прогресс-бара (для режима чанков)"""
        try:
            if value is not None:
                self.progress['value'] = value
        except Exception as e:
            logging.error(f"Ошибка при обновлении прогресс-бара: {e}")
    
    def stop_progress(self):
        """Остановка прогресс-бара"""
        try:
            self.progress['value'] = self.progress['maximum']
        except Exception as e:
            logging.error(f"Ошибка при остановке прогресс-бара: {e}")
    
    def show_error(self, title, message):
        """Показ сообщения об ошибке"""
        try:
            self.root.after(0, lambda: messagebox.showerror(title, message))
        except Exception as e:
            logging.error(f"Ошибка при показе сообщения об ошибке: {e}")
    
    def show_warning(self, title, message):
        """Показ предупреждения"""
        try:
            self.root.after(0, lambda: messagebox.showwarning(title, message))
        except Exception as e:
            logging.error(f"Ошибка при показе предупреждения: {e}")
    
    def show_info(self, title, message):
        """Показ информационного сообщения"""
        try:
            self.root.after(0, lambda: messagebox.showinfo(title, message))
        except Exception as e:
            logging.error(f"Ошибка при показе информации: {e}")
    
    def play_audio_threaded(self):
        """Запуск генерации и воспроизведения в отдельном потоке"""
        if not self.is_model_loaded:
            self.show_warning("Внимание", "Модель ещё загружается. Пожалуйста, подождите.")
            return
        
        logging.info("Запуск потока для воспроизведения аудио")
        thread = threading.Thread(target=self.play_audio, daemon=True)
        thread.start()
    
    def play_audio(self):
        """Генерация и воспроизведение аудио"""
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            logging.warning("Попытка воспроизведения пустого текста")
            self.show_warning("Внимание", "Введите текст для озвучки")
            return
        
        speaker = self.speaker_combo.get()
        logging.info(f"=== Начало воспроизведения: текст='{text[:50]}...', голос='{speaker}' ===")
        
        try:
            # Сброс флага остановки перед началом генерации
            self.reset_stop_flag()

            text = self._preprocess_text_for_tts(text)
            
            self.update_status(f"Статус: Генерация речи голосом '{speaker}'...")
            self.start_progress()

            chunk_mode, _, max_chars, silence_ms = self._get_chunk_settings()
            use_chunking = chunk_mode or self._should_use_chunking(text)
            
            # Получаем текущую скорость озвучки
            speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'

            if use_chunking:
                logging.info("Включён режим чанков для воспроизведения")
                audio_int16, _chunks = self.generate_audio_chunked(text, speaker, max_chars, silence_ms)
            else:
                audio = self.generate_audio(text, speaker, speech_rate)
                audio_int16 = self._tensor_audio_to_int16_mono(audio)

            logging.info("Аудио успешно сгенерировано")
            
            self.update_status("Статус: Воспроизведение...")
            
            logging.debug(f"Аудио int16: длина={len(audio_int16)}, диапазон: [{audio_int16.min()}, {audio_int16.max()}]")
            
            # *** ИСПРАВЛЕНИЕ: Преобразование для стерео-микшера ***
            # Проверяем размерность и преобразуем для совместимости с Pygame
            if len(audio_int16.shape) == 1:
                # Для моно аудио создаем 2-мерный массив с двумя одинаковыми каналами (стерео)
                audio_for_pygame = np.column_stack((audio_int16, audio_int16))
                logging.info(f"Аудио преобразовано из 1D в 2D стерео: {audio_for_pygame.shape}")
            elif len(audio_int16.shape) == 2 and audio_int16.shape[1] == 1:
                # Если аудио уже 2D с одним каналом, дублируем для стерео
                audio_for_pygame = np.column_stack((audio_int16[:, 0], audio_int16[:, 0]))
                logging.info("Аудио преобразовано из моно в стерео")
            elif len(audio_int16.shape) == 2 and audio_int16.shape[1] > 1:
                # Если аудио уже стерео, оставляем как есть
                audio_for_pygame = audio_int16
                logging.info("Аудио уже в стерео формате")
            else:
                audio_for_pygame = audio_int16
                logging.info("Аудио в неизвестном формате, оставляем как есть")
            
            # Проверяем инициализацию микшера
            if not pygame.mixer.get_init():
                logging.info("Микшер не инициализирован, выполняем повторную инициализацию")
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
            
            # Останавливаем предыдущее воспроизведение
            self.stop_audio()
            
            # Создаем и воспроизводим новый звук
            self.current_sound = pygame.sndarray.make_sound(audio_for_pygame)
            logging.info("Аудио загружено в pygame, начинается воспроизведение")
            
            duration = len(audio_int16) / SAMPLE_RATE
            logging.info(f"Длительность аудио: {duration:.2f} секунд")
            
            self.current_sound.play()
            
            # Ждем окончания воспроизведения
            while pygame.mixer.get_busy():
                pygame.time.wait(100)
            
            logging.info("Воспроизведение аудио завершено")
            self.update_status(f"Статус: Готово (голос: {speaker}) ✅")
            self.stop_progress()
            
        except InterruptedError:
            # Обработка остановки пользователем
            logging.info("Воспроизведение остановлено пользователем")
            self.stop_progress()
            
        except Exception as e:
            logging.error(f"Ошибка при воспроизведении аудио: {str(e)}", exc_info=True)
            self.update_status("Статус: Ошибка ❌")
            self.show_error("Ошибка", str(e))
            self.stop_progress()
    
    def stop_audio(self):
        """Остановка воспроизведения и генерации"""
        try:
            # Остановка воспроизведения
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                if self.current_sound:
                    self.current_sound.stop()
                    self.current_sound = None
                self.update_status("Статус: Воспроизведение остановлено")
                logging.info("Воспроизведение остановлено пользователем")
            
            # Установка флага остановки генерации
            self.stop_generation_flag = True
            logging.info("Установлен флаг остановки генерации")
            
        except Exception as e:
            logging.error(f"Ошибка при остановке: {e}")
    
    def reset_stop_flag(self):
        """Сброс флага остановки перед началом новой генерации"""
        self.stop_generation_flag = False
        logging.debug("Флаг остановки генерации сброшен")
    
    def check_stop_flag(self):
        """Проверка флага остановки. Вызывает исключение при остановке."""
        if self.stop_generation_flag:
            logging.info("Генерация остановлена пользователем")
            self.update_status("Статус: Генерация остановлена пользователем")
            self.stop_progress()
            raise InterruptedError("Генерация остановлена пользователем")
    
    def save_audio_threaded(self):
        """Запуск сохранения аудио в отдельном потоке"""
        if not self.is_model_loaded:
            self.show_warning("Внимание", "Модель ещё загружается. Пожалуйста, подождите.")
            return
        
        logging.info("Запуск потока для сохранения аудио")
        thread = threading.Thread(target=self.save_audio, daemon=True)
        thread.start()
    
    def save_audio(self):
        """Генерация и сохранение аудио в файл WAV"""
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            logging.warning("Попытка сохранения пустого текста")
            self.show_warning("Внимание", "Введите текст для озвучки")
            return
        
        speaker = self.speaker_combo.get()
        logging.info(f"=== Начало сохранения аудио: текст='{text[:50]}...', голос='{speaker}' ===")
        
        try:
            # Сброс флага остановки перед началом генерации
            self.reset_stop_flag()

            text = self._preprocess_text_for_tts(text)
            
            self.update_status(f"Статус: Генерация для сохранения...")
            self.start_progress()
            
            # Получение целевой директории
            target_dir = self.target_dir_var.get() if hasattr(self, 'target_dir_var') else AUDIO_DIR
            
            # Создание директории для аудио, если не существует
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                logging.info(f"Создана директория для аудио: {target_dir}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_dir = self.chunk_dir_var.get().strip() if hasattr(self, 'chunk_dir_var') else "my_audiobook"
            if not custom_dir:
                custom_dir = "my_audiobook"
            base_name = f"{custom_dir}_{timestamp}"

            chunk_mode, save_parts, max_chars, silence_ms = self._get_chunk_settings()
            use_chunking = chunk_mode or self._should_use_chunking(text)

            if not use_chunking:
                filename = f"{base_name}.wav"
                full_path = os.path.join(target_dir, filename)
                logging.info(f"Сохранение в файл: {full_path}")

                self.model.save_wav(
                    text=text,
                    speaker=speaker,
                    sample_rate=SAMPLE_RATE,
                    audio_path=full_path
                )
            else:
                logging.info("Включён режим чанков для сохранения")
                chunks = self.split_text_into_chunks(text, max_chars)
                
                # Запускаем прогресс-бар с количеством чанков
                self.start_progress(total_chunks=len(chunks))
                
                silence_samples = int(SAMPLE_RATE * (silence_ms / 1000.0))
                silence = np.zeros((silence_samples,), dtype=np.int16) if silence_samples > 0 else None
                
                audio_parts = []
                for idx, chunk_text in enumerate(chunks, start=1):
                    # Проверка флага остановки
                    self.check_stop_flag()
                    
                    self.update_status(f"Статус: Генерация части {idx}/{len(chunks)}...")
                    speech_rate = self.speech_rate_var.get() if hasattr(self, 'speech_rate_var') else 'medium'
                    audio_tensor = self.generate_audio(chunk_text, speaker, speech_rate)
                    audio_part_int16 = self._tensor_audio_to_int16_mono(audio_tensor)
                    audio_parts.append(audio_part_int16)
                    
                    # Сохраняем часть если нужно
                    if save_parts:
                        parts_dir = os.path.join(target_dir, f"{base_name}_parts")
                        os.makedirs(parts_dir, exist_ok=True)
                        logging.info(f"Сохранение частей в: {parts_dir}")
                        part_filename = f"part_{idx:03d}.wav"
                        part_path = os.path.join(parts_dir, part_filename)
                        self._write_wav_int16_mono(part_path, audio_part_int16)
                    
                    if silence is not None and idx != len(chunks):
                        audio_parts.append(silence)
                    
                    # Обновляем прогресс-бар
                    self.update_progress(idx)

                filename = f"{base_name}.wav"
                full_path = os.path.join(target_dir, filename)
                logging.info(f"Сохранение итогового WAV: {full_path}")
                audio_full_int16 = np.concatenate(audio_parts) if audio_parts else np.zeros((0,), dtype=np.int16)
                self._write_wav_int16_mono(full_path, audio_full_int16)

            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path) / 1024
                logging.info(f"Файл {full_path} успешно создан (размер: {file_size:.2f} КБ)")
                
                # Конвертация в MP3 если включена опция
                convert_to_mp3 = bool(self.convert_to_mp3_var.get()) if hasattr(self, 'convert_to_mp3_var') else False
                mp3_path = None
                if convert_to_mp3:
                    self.update_status(f"Статус: Конвертация в MP3...")
                    bitrate = self.mp3_bitrate_var.get() if hasattr(self, 'mp3_bitrate_var') else "192k"
                    try:
                        mp3_path = self.convert_wav_to_mp3(full_path, bitrate)
                        logging.info(f"MP3 файл создан: {mp3_path}")
                        
                        # Удаляем WAV файл после успешной конвертации в MP3
                        try:
                            os.remove(full_path)
                            logging.info(f"WAV файл удалён: {full_path}")
                        except Exception as remove_error:
                            logging.warning(f"Не удалось удалить WAV файл: {remove_error}")
                        
                        # Удаляем директорию с частями если включена опция
                        delete_wav_dir = bool(self.delete_wav_dir_var.get()) if hasattr(self, 'delete_wav_dir_var') else False
                        if delete_wav_dir and use_chunking and save_parts:
                            parts_dir = os.path.join(target_dir, f"{base_name}_parts")
                            logging.info(f"=== ПРОВЕРКА УДАЛЕНИЯ ДИРЕКТОРИИ С ЧАСТЯМИ ===")
                            logging.info(f"  delete_wav_dir: {delete_wav_dir}")
                            logging.info(f"  use_chunking: {use_chunking}")
                            logging.info(f"  save_parts: {save_parts}")
                            logging.info(f"  parts_dir: {parts_dir}")
                            logging.info(f"  parts_dir существует: {os.path.exists(parts_dir)}")
                            try:
                                import shutil
                                if os.path.exists(parts_dir):
                                    shutil.rmtree(parts_dir)
                                    logging.info(f"Директория с частями удалена: {parts_dir}")
                                    logging.info(f"parts_dir существует после удаления: {os.path.exists(parts_dir)}")
                                else:
                                    logging.warning(f"Директория с частями не найдена: {parts_dir}")
                            except Exception as del_error:
                                logging.error(f"Не удалось удалить директорию с частями: {del_error}", exc_info=True)
                    except Exception as mp3_error:
                        logging.error(f"Ошибка конвертации в MP3: {mp3_error}")
                        self.show_warning("Предупреждение", f"WAV сохранён, но конвертация в MP3 не удалась:\n{str(mp3_error)}")
                
                if mp3_path:
                    mp3_size = os.path.getsize(mp3_path) / 1024
                    self.update_status(f"Статус: Сохранено в MP3 ✅")
                    extra = ""
                    if use_chunking and save_parts:
                        extra = f"\n\nЧасти сохранены в папку:\n{os.path.join(target_dir, f'{base_name}_parts')}"
                    message = f"Аудио сохранено в MP3 файл:\n{mp3_path}\n\nРазмер: {mp3_size:.2f} КБ, битрейт: {bitrate}{extra}"
                    # WAV файл был удалён
                else:
                    self.update_status(f"Статус: Сохранено в {os.path.basename(full_path)} ✅")
                    extra = ""
                    if use_chunking and save_parts:
                        extra = f"\n\nЧасти сохранены в папку:\n{os.path.join(target_dir, f'{base_name}_parts')}"
                    message = f"Аудио сохранено в файл:\n{full_path}\n\nРазмер: {file_size:.2f} КБ{extra}"
                
                self.show_info("Успех", message)
            else:
                logging.error(f"Файл {full_path} не был создан")
                raise Exception("Файл не был создан")
            
            self.stop_progress()
            
        except InterruptedError:
            # Обработка остановки пользователем
            logging.info("Сохранение аудио остановлено пользователем")
            self.update_status("Статус: Сохранение остановлено пользователем")
            self.stop_progress()
            
        except Exception as e:
            logging.error(f"Ошибка при сохранении аудио: {str(e)}", exc_info=True)
            self.update_status("Статус: Ошибка сохранения ❌")
            self.show_error("Ошибка", str(e))
            self.stop_progress()
    
    def convert_wav_to_mp3(self, wav_path, bitrate="192k"):
        """Конвертация WAV файла в MP3 с указанным битрейтом"""
        try:
            logging.info(f"Конвертация WAV в MP3: {wav_path} (битрейт: {bitrate})")
            
            # Загрузка WAV файла
            audio = AudioSegment.from_wav(wav_path)
            
            # Создание пути для MP3 файла
            mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
            
            # Экспорт в MP3 с указанным битрейтом
            audio.export(mp3_path, format="mp3", bitrate=bitrate)
            
            if os.path.exists(mp3_path):
                mp3_size = os.path.getsize(mp3_path) / 1024
                logging.info(f"MP3 файл создан: {mp3_path} (размер: {mp3_size:.2f} КБ)")
                return mp3_path
            else:
                raise Exception("MP3 файл не был создан")
                
        except Exception as e:
            logging.error(f"Ошибка при конвертации в MP3: {str(e)}", exc_info=True)
            raise
    
    def show_find_dialog(self, text_widget):
        """Показ диалога поиска"""
        # Создаем топ-уровневое окно поиска
        find_dialog = tk.Toplevel(self.root)
        find_dialog.title("Поиск")
        find_dialog.geometry("300x100")
        find_dialog.transient(self.root)
        find_dialog.grab_set()
        
        # Фрейм для элементов
        frame = ttk.Frame(find_dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Поле ввода для поиска
        ttk.Label(frame, text="Найти:").grid(row=0, column=0, sticky=tk.W, pady=5)
        search_entry = ttk.Entry(frame, width=30)
        search_entry.grid(row=0, column=1, padx=5, pady=5)
        search_entry.focus_set()
        
        # Переменная для хранения последнего найденной позиции
        search_state = {'last_pos': '1.0', 'search_text': ''}
        
        def do_find():
            """Выполнение поиска"""
            search_text = search_entry.get().strip()
            if not search_text:
                return
            
            search_state['search_text'] = search_text
            search_state['last_pos'] = '1.0'
            
            # Ищем первое вхождение
            pos = text_widget.search(search_text, '1.0', stopindex=tk.END)
            if pos:
                # Выделяем найденный текст
                end_pos = f"{pos}+{len(search_text)}c"
                text_widget.tag_remove('search', '1.0', tk.END)
                text_widget.tag_add('search', pos, end_pos)
                text_widget.tag_config('search', background='yellow', foreground='black')
                text_widget.mark_set('insert', end_pos)
                text_widget.see(pos)
                search_state['last_pos'] = end_pos
            else:
                messagebox.showinfo("Поиск", "Текст не найден", parent=find_dialog)
                search_state['last_pos'] = '1.0'
        
        def find_next_occurrence():
            """Поиск следующего вхождения"""
            search_text = search_state.get('search_text', '')
            if not search_text:
                # Если еще не искали, берем из поля ввода
                search_text = search_entry.get().strip()
                if not search_text:
                    return
                search_state['search_text'] = search_text
                search_state['last_pos'] = '1.0'
            
            # Ищем следующее вхождение
            pos = text_widget.search(search_text, search_state['last_pos'], stopindex=tk.END)
            if pos:
                # Выделяем найденный текст
                end_pos = f"{pos}+{len(search_text)}c"
                text_widget.tag_remove('search', '1.0', tk.END)
                text_widget.tag_add('search', pos, end_pos)
                text_widget.tag_config('search', background='yellow', foreground='black')
                text_widget.mark_set('insert', end_pos)
                text_widget.see(pos)
                search_state['last_pos'] = end_pos
            else:
                # Если не найдено, начинаем с начала
                pos = text_widget.search(search_text, '1.0', stopindex=search_state['last_pos'])
                if pos:
                    end_pos = f"{pos}+{len(search_text)}c"
                    text_widget.tag_remove('search', '1.0', tk.END)
                    text_widget.tag_add('search', pos, end_pos)
                    text_widget.tag_config('search', background='yellow', foreground='black')
                    text_widget.mark_set('insert', end_pos)
                    text_widget.see(pos)
                    search_state['last_pos'] = end_pos
                else:
                    messagebox.showinfo("Поиск", "Больше не найдено", parent=find_dialog)
                    search_state['last_pos'] = '1.0'
        
        # Кнопки
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Найти", command=do_find).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Далее", command=find_next_occurrence).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Закрыть", command=find_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Обработка Enter в поле поиска
        search_entry.bind('<Return>', lambda e: do_find())
        
        # Сохраняем ссылку на диалог для закрытия при повторном Ctrl+F
        self._find_dialog = find_dialog
    
    def find_text(self, text_widget):
        """Запуск поиска для текстового поля"""
        # Закрываем предыдущий диалог если есть
        if hasattr(self, '_find_dialog'):
            try:
                self._find_dialog.destroy()
            except:
                pass
        self.show_find_dialog(text_widget)
    
    def find_next(self, text_widget):
        """Поиск следующего вхождения"""
        if hasattr(self, '_find_dialog'):
            try:
                # Получаем состояние поиска из диалога
                for widget in self._find_dialog.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Entry):
                                search_text = child.get().strip()
                                if search_text:
                                    # Ищем следующее вхождение
                                    try:
                                        # Получаем последнюю позицию из тега search
                                        try:
                                            last_pos = text_widget.tag_ranges('search')[1].string
                                        except:
                                            last_pos = '1.0'
                                        
                                        pos = text_widget.search(search_text, last_pos, stopindex=tk.END)
                                        if pos:
                                            end_pos = f"{pos}+{len(search_text)}c"
                                            text_widget.tag_remove('search', '1.0', tk.END)
                                            text_widget.tag_add('search', pos, end_pos)
                                            text_widget.tag_config('search', background='yellow', foreground='black')
                                            text_widget.mark_set('insert', end_pos)
                                            text_widget.see(pos)
                                        else:
                                            # Начинаем с начала
                                            pos = text_widget.search(search_text, '1.0', stopindex=last_pos)
                                            if pos:
                                                end_pos = f"{pos}+{len(search_text)}c"
                                                text_widget.tag_remove('search', '1.0', tk.END)
                                                text_widget.tag_add('search', pos, end_pos)
                                                text_widget.tag_config('search', background='yellow', foreground='black')
                                                text_widget.mark_set('insert', end_pos)
                                                text_widget.see(pos)
                                            else:
                                                messagebox.showinfo("Поиск", "Больше не найдено")
                                    except Exception as e:
                                        logging.error(f"Ошибка при поиске: {e}")
                                break
            except Exception as e:
                logging.error(f"Ошибка при поиске следующего: {e}")
        else:
            # Если диалог не открыт, показываем его
            self.show_find_dialog(text_widget)
    
    def copy_to_clipboard(self, text_widget):
        """Копирование выделенного текста в буфер обмена"""
        try:
            selected_text = text_widget.get("sel.first", "sel.last")
            if selected_text:
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
                self.root.update()  # Обновляем буфер обмена
                logging.debug(f"Текст скопирован в буфер обмена ({len(selected_text)} символов)")
            else:
                # Если нет выделения, копируем весь текст
                all_text = text_widget.get("1.0", tk.END).strip()
                if all_text:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(all_text)
                    self.root.update()
                    logging.debug(f"Весь текст скопирован в буфер обмена ({len(all_text)} символов)")
        except tk.TclError:
            # Нет выделенного текста
            pass
        except Exception as e:
            logging.error(f"Ошибка при копировании в буфер обмена: {e}")
    
    def paste_from_clipboard(self, text_widget):
        """Вставка текста из буфера обмена"""
        try:
            clipboard_text = self.root.clipboard_get()
            if clipboard_text:
                # Вставляем в позицию курсора или заменяем выделение
                try:
                    # Если есть выделение, заменяем его
                    text_widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    # Нет выделения, вставляем в позицию курсора
                    pass
                text_widget.insert("insert", clipboard_text)
                logging.debug(f"Текст вставлен из буфера обмена ({len(clipboard_text)} символов)")
        except tk.TclError:
            # Буфер обмена пуст или недоступен
            pass
        except Exception as e:
            logging.error(f"Ошибка при вставке из буфера обмена: {e}")
        # Возвращаем 'break' чтобы предотвратить стандартную вставку
        return "break"
    
    def cut_to_clipboard(self, text_widget):
        """Вырезание выделенного текста в буфер обмена"""
        try:
            selected_text = text_widget.get("sel.first", "sel.last")
            if selected_text:
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
                self.root.update()
                text_widget.delete("sel.first", "sel.last")
                logging.debug(f"Текст вырезан в буфер обмена ({len(selected_text)} символов)")
        except tk.TclError:
            # Нет выделенного текста
            pass
        except Exception as e:
            logging.error(f"Ошибка при вырезании в буфер обмена: {e}")
    
    def clear_log(self):
        """Очистка лога (информационное сообщение)"""
        logging.info("Очистка лога вызвана - логи теперь только в консоли")
        self.show_info("Лог", "Логирование теперь выполняется только в консоль.\nФайлы логов не создаются.")
    
    def open_audio_folder(self):
        """Открытие папки с аудиофайлами в проводнике Windows"""
        try:
            # Получение целевой директории: приоритет - переменная, затем конфиг, затем AUDIO_DIR
            if hasattr(self, 'target_dir_var'):
                raw_target = self.target_dir_var.get()
                source = "target_dir_var"
            elif hasattr(self, 'saved_config') and 'target_dir' in self.saved_config:
                raw_target = self.saved_config['target_dir']
                source = "saved_config"
            else:
                raw_target = AUDIO_DIR
                source = "AUDIO_DIR"
            
            # Нормализация пути для Windows (конвертация слэшей)
            target_dir = os.path.normpath(raw_target)
            
            # Детальное логирование для отладки
            logging.info(f"Источник пути: {source}")
            logging.info(f"Сырой путь: {raw_target}")
            logging.info(f"Нормализованный путь: {target_dir}")
            logging.info(f"Путь существует: {os.path.exists(target_dir)}")
            
            # Создание директории, если не существует
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                logging.info(f"Создана директория для аудио: {target_dir}")
            
            # Открытие папки в проводнике Windows через os.startfile (более надёжно)
            os.startfile(target_dir)
            logging.info(f"Открыта папка с аудио: {target_dir}")
            self.update_status(f"Статус: Открыта папка: {target_dir} ✅")
        except Exception as e:
            logging.error(f"Ошибка при открытии папки: {e}", exc_info=True)
            # Повторное получение директории для отображения в ошибке
            if hasattr(self, 'target_dir_var'):
                raw_target = self.target_dir_var.get()
            elif hasattr(self, 'saved_config') and 'target_dir' in self.saved_config:
                raw_target = self.saved_config['target_dir']
            else:
                raw_target = AUDIO_DIR
            target_dir = os.path.normpath(raw_target)
            self.show_error("Ошибка", f"Не удалось открыть папку:\n{target_dir}\n\nОшибка: {e}")
    
    def cleanup(self):
        """Очистка ресурсов при закрытии"""
        logging.info("Очистка ресурсов перед закрытием")
        try:
            self.stop_audio()
            pygame.mixer.quit()
            logging.info("Ресурсы pygame освобождены")
        except Exception as e:
            logging.error(f"Ошибка при очистке ресурсов: {e}")
        
    def on_closing(self):
        """Обработка закрытия окна"""
        logging.info("Приложение закрывается")
        self.save_config()
        self.cleanup()
        self.root.destroy()
        logging.shutdown()

def run_cli(args):
    """Запуск TTS в режиме командной строки (без GUI)"""
    logging.info("=== ЗАПУСК В CLI РЕЖИМЕ ===")
    
    try:
        # Инициализация препроцессора
        text_preprocessor = TextPreprocessor()
        
        # Загрузка текста из файла или из аргумента
        if args.input_file:
            logging.info(f"Загрузка текста из файла: {args.input_file}")
            import zipfile
            import xml.etree.ElementTree as ET
            
            file_ext = os.path.splitext(args.input_file)[1].lower()
            
            if file_ext == '.zip':
                # Поиск FB2 файла внутри ZIP архива
                logging.info("Распаковка FB2 из ZIP архива...")
                with zipfile.ZipFile(args.input_file, 'r') as zf:
                    # Попытка получить имена файлов в правильной кодировке
                    try:
                        # Сначала пробуем UTF-8
                        fb2_files = [f for f in zf.namelist() if f.endswith('.fb2')]
                    except UnicodeDecodeError:
                        # Если не получилось, используем CP866 (для кириллицы)
                        fb2_files = [f for f in zf.namelist() if f.endswith('.fb2')]
                    
                    if not fb2_files:
                        logging.error("FB2 файл не найден в ZIP архиве")
                        print("Ошибка: FB2 файл не найден в ZIP архиве")
                        return 1
                    
                    fb2_file = fb2_files[0]
                    logging.info(f"Найден FB2 файл: {fb2_file}")
                    
                    # Чтение содержимого FB2 файла
                    with zf.open(fb2_file) as f:
                        fb2_bytes = f.read()
                        # Пробуем UTF-8, если не получается - CP1251
                        try:
                            fb2_content = fb2_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            fb2_content = fb2_bytes.decode('cp1251')
                
                # Парсинг FB2 содержимого
                root = ET.fromstring(fb2_content)
                text_parts = []
                for elem in root.iter():
                    if elem.tag.endswith('p') or elem.tag.endswith('title'):
                        if elem.text:
                            text_parts.append(elem.text.strip())
                text = '\n'.join(text_parts)
                logging.info(f"Текст извлечён из FB2 в ZIP (длина: {len(text)})")
                
            elif file_ext == '.fb2':
                # Парсинг FB2 файла
                logging.info("Парсинг FB2 файла...")
                tree = ET.parse(args.input_file)
                root = tree.getroot()
                
                text_parts = []
                for elem in root.iter():
                    if elem.tag.endswith('p') or elem.tag.endswith('title'):
                        if elem.text:
                            text_parts.append(elem.text.strip())
                text = '\n'.join(text_parts)
                logging.info(f"Текст извлечён из FB2 (длина: {len(text)})")
                
            else:
                # Чтение TXT файла
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                logging.info(f"Текст загружен из TXT (длина: {len(text)})")
        elif args.text:
            text = args.text
        else:
            logging.error("Не указан текст или файл для озвучки")
            print("Ошибка: Необходимо указать --text или --input-file")
            return 1
        
        if not text.strip():
            logging.error("Пустой текст")
            print("Ошибка: Пустой текст")
            return 1
        
        logging.info(f"Текст для озвучки: {len(text)} символов")
        
        # Предобработка текста если включена
        if args.preprocess:
            logging.info("Выполняется предобработка текста...")
            use_ruaccent = args.ruaccent if hasattr(args, 'ruaccent') else False
            text = text_preprocessor.preprocess(
                text,
                use_num2words=not args.no_num2words,
                use_ruaccent=use_ruaccent
            )
            logging.info(f"Текст предобработан: {len(text)} символов")
        
        # Загрузка модели
        logging.info(f"Загрузка модели TTS из: {MODEL_FILE}")
        device = torch.device('cpu')
        torch.set_num_threads(args.threads if hasattr(args, 'threads') else 4)
        
        if not os.path.exists(MODEL_FILE):
            logging.info("Модель не найдена, начинается скачивание...")
            os.makedirs(CACHE_DIR, exist_ok=True)
            torch.hub.download_url_to_file(MODEL_URL, MODEL_FILE, progress=True)
        
        model = torch.package.PackageImporter(MODEL_FILE).load_pickle("tts_models", "model")
        model.to(device)
        logging.info("Модель загружена")
        
        # Получение параметров
        speaker = args.speaker if args.speaker else 'baya'
        speech_rate = args.speech_rate if hasattr(args, 'speech_rate') else 'medium'
        max_chars = args.max_chars if hasattr(args, 'max_chars') else DEFAULT_MAX_CHARS_PER_CHUNK
        silence_ms = args.silence_ms if hasattr(args, 'silence_ms') else DEFAULT_SILENCE_MS
        convert_to_mp3 = args.mp3 if hasattr(args, 'mp3') else False
        mp3_bitrate = args.bitrate if hasattr(args, 'bitrate') else '192k'
        
        # Определение целевой директории
        output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Целевая директория: {output_dir}")
        
        # Определение выходного файла
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tts_output_{timestamp}.wav"
            output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename
        elif output_dir and not os.path.isabs(output_path):
            output_path = os.path.join(output_dir, output_path)
        
        # Разбиение на чанки если текст большой или включен режим чанков
        use_chunking = args.chunks if hasattr(args, 'chunks') else len(text) > AUTO_CHUNK_THRESHOLD
        
        if use_chunking:
            logging.info(f"Режим чанков: max_chars={max_chars}, silence={silence_ms}ms")
            chunks = text_preprocessor._split_text_into_chunks(text, max_chars)
            logging.info(f"Текст разбит на {len(chunks)} чанков")
            
            # Создание директории для частей если включена опция
            save_parts = args.save_parts if hasattr(args, 'save_parts') else False
            parts_dir = None
            if save_parts:
                parts_dir = os.path.splitext(output_path)[0] + "_parts"
                os.makedirs(parts_dir, exist_ok=True)
                logging.info(f"Директория для частей создана: {parts_dir}")
            
            silence_samples = int(SAMPLE_RATE * (silence_ms / 1000.0))
            silence = np.zeros((silence_samples,), dtype=np.int16) if silence_samples > 0 else None
            
            audio_parts = []
            for idx, chunk_text in enumerate(chunks, start=1):
                logging.info(f"Обработка чанка {idx}/{len(chunks)}")
                
                # Формирование SSML
                chunk_escaped = html.escape(chunk_text, quote=False)
                # Добавляем теги пауз <s> вокруг знаков препинания для валидного XML
                chunk_with_pauses = re.sub(r'\.\s*', '<s>.</s> ', chunk_escaped)
                chunk_with_pauses = re.sub(r'\?\s*', '<s>?</s> ', chunk_with_pauses)
                chunk_with_pauses = re.sub(r'!\s*', '<s>!</s> ', chunk_with_pauses)
                chunk_with_pauses = re.sub(r'…\s*', '<s>…</s> ', chunk_with_pauses)
                chunk_with_pauses = ' '.join(chunk_with_pauses.split())
                ssml_text = f'<speak><prosody rate="{speech_rate}">{chunk_with_pauses}</prosody></speak>'
                
                audio_tensor = model.apply_tts(
                    ssml_text=ssml_text,
                    speaker=speaker,
                    sample_rate=SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True
                )
                audio_int16 = (audio_tensor.numpy() * 32767).astype(np.int16)
                if len(audio_int16.shape) == 2:
                    audio_int16 = audio_int16[:, 0]
                
                # Сохранение части в файл если включена опция
                if save_parts and parts_dir:
                    part_path = os.path.join(parts_dir, f"part_{idx:03d}.wav")
                    with wave.open(part_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_int16.tobytes())
                    logging.debug(f"Часть {idx} сохранена: {part_path}")
                
                audio_parts.append(audio_int16)
                if silence is not None and idx != len(chunks):
                    audio_parts.append(silence)
            
            audio_full = np.concatenate(audio_parts)
        else:
            logging.info("Генерация без разбиения на чанки")
            text_escaped = html.escape(text, quote=False)
            # Добавляем теги пауз <s> вокруг знаков препинания для валидного XML
            text_with_pauses = re.sub(r'\.\s*', '<s>.</s> ', text_escaped)
            text_with_pauses = re.sub(r'\?\s*', '<s>?</s> ', text_with_pauses)
            text_with_pauses = re.sub(r'!\s*', '<s>!</s> ', text_with_pauses)
            text_with_pauses = re.sub(r'…\s*', '<s>…</s> ', text_with_pauses)
            text_with_pauses = ' '.join(text_with_pauses.split())
            ssml_text = f'<speak><prosody rate="{speech_rate}">{text_with_pauses}</prosody></speak>'
            
            audio_tensor = model.apply_tts(
                ssml_text=ssml_text,
                speaker=speaker,
                sample_rate=SAMPLE_RATE,
                put_accent=True,
                put_yo=True
            )
            audio_full = (audio_tensor.numpy() * 32767).astype(np.int16)
            if len(audio_full.shape) == 2:
                audio_full = audio_full[:, 0]
        
        # Сохранение WAV
        logging.info(f"Сохранение WAV: {output_path}")
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_full.tobytes())
        
        wav_size = os.path.getsize(output_path) / 1024
        logging.info(f"WAV сохранён: {output_path} ({wav_size:.2f} КБ)")
        
        # Конвертация в MP3 если включена
        if convert_to_mp3:
            logging.info(f"Конвертация в MP3 (битрейт: {mp3_bitrate})...")
            audio = AudioSegment.from_wav(output_path)
            mp3_path = os.path.splitext(output_path)[0] + ".mp3"
            audio.export(mp3_path, format="mp3", bitrate=mp3_bitrate)
            mp3_size = os.path.getsize(mp3_path) / 1024
            logging.info(f"MP3 сохранён: {mp3_path} ({mp3_size:.2f} КБ)")
            
            # Удаление WAV если запрошено
            if args.no_wav if hasattr(args, 'no_wav') else False:
                os.remove(output_path)
                logging.info(f"WAV удалён: {output_path}")
                output_path = mp3_path
            
            # Удаление директории с частями если запрошено
            delete_parts = args.delete_parts if hasattr(args, 'delete_parts') else False
            if delete_parts and parts_dir and os.path.exists(parts_dir):
                try:
                    import shutil
                    shutil.rmtree(parts_dir)
                    logging.info(f"Директория с частями удалена: {parts_dir}")
                except Exception as del_error:
                    logging.error(f"Не удалось удалить директорию с частями: {del_error}")
        
        print(f"✅ Аудио сохранено: {output_path}")
        logging.info("=== CLI режим завершен успешно ===")
        return 0
        
    except Exception as e:
        logging.error(f"Ошибка в CLI режиме: {e}", exc_info=True)
        print(f"❌ Ошибка: {e}")
        return 1


def main():
    """Главная функция запуска приложения"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Silero TTS - синтез речи из текста в MP3/WAV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python text2mp3.py --text "Привет мир!" -o output.wav
  python text2mp3.py -i input.txt -o output.mp3 --mp3
  python text2mp3.py --text "Длинный текст..." --chunks --max-chars 1000
  python text2mp3.py  # запуск GUI интерфейса
        '''
    )
    
    # Режим работы
    parser.add_argument('--gui', action='store_true', help='Запуск GUI интерфейса (по умолчанию)')
    
    # Входные данные
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', '-t', type=str, help='Текст для озвучки')
    input_group.add_argument('--input-file', '-i', type=str, help='Файл с текстом (txt, fb2)')
    
    # Выходные данные
    parser.add_argument('--output', '-o', type=str, help='Выходной файл (WAV или MP3)')
    parser.add_argument('--mp3', action='store_true', help='Конвертировать в MP3')
    parser.add_argument('--bitrate', type=str, default='192k', help='Битрейт MP3 (128k, 192k, 256k, 320k)')
    parser.add_argument('--no-wav', action='store_true', help='Удалить WAV после конвертации в MP3')
    
    # Параметры синтеза
    parser.add_argument('--speaker', '-s', type=str, choices=SPEAKERS, default='baya', help='Голос диктора')
    parser.add_argument('--speech-rate', '-r', type=str, choices=['x-slow', 'slow', 'medium', 'fast', 'x-fast'], default='medium', help='Скорость речи')
    parser.add_argument('--threads', type=int, default=4, help='Количество потоков CPU')
    
    # Параметры чанков
    parser.add_argument('--chunks', action='store_true', help='Разбивать текст на чанки')
    parser.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK, help='Макс. символов в чанке')
    parser.add_argument('--silence-ms', type=int, default=DEFAULT_SILENCE_MS, help='Пауза между чанками (мс)')
    parser.add_argument('--save-parts', action='store_true', help='Сохранять чанки в отдельную директорию')
    parser.add_argument('--delete-parts', action='store_true', help='Удалить директорию с частями после конвертации в MP3')
    parser.add_argument('--output-dir', type=str, help='Целевая директория для сохранения аудио')
    
    # Предобработка
    parser.add_argument('--preprocess', action='store_true', help='Использовать предобработку текста')
    parser.add_argument('--no-num2words', action='store_true', help='Не заменять числа словами')
    parser.add_argument('--ruaccent', action='store_true', help='Использовать ruaccent для ударений')
    
    # Отладка
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод логов')
    parser.add_argument('--quiet', '-q', action='store_true', help='Не выводить логи (только ошибки)')
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Если указан --gui или нет аргументов ввода - запускаем GUI
    if args.gui or (not args.text and not args.input_file):
        logging.info("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
        
        try:
            root = tk.Tk()
            app = SileroTTSApp(root)
            root.mainloop()
        except Exception as e:
            logging.critical(f"Критическая ошибка при запуске приложения: {e}", exc_info=True)
            print(f"Критическая ошибка: {e}")
            sys.exit(1)
        finally:
            logging.info("=== ПРИЛОЖЕНИЕ ЗАВЕРШЕНО ===")
    else:
        # Запуск CLI режима
        sys.exit(run_cli(args))

if __name__ == "__main__":
    main()
