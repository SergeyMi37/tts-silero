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

# Настройка логирования
log_filename = f'tts_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Константы
SAMPLE_RATE = 48000
MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_ru.pt'
MODEL_FILE = 'v5_ru.pt'

# Доступные русские голоса из Silero v5
SPEAKERS = ['baya', 'eugene', 'kseniya', 'xenia', 'random']

class SileroTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Silero TTS Озвучка текста (v5)")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.model = None
        self.is_model_loaded = False
        self.current_sound = None  # Для хранения текущего звукового объекта
        
        logging.info("Инициализация приложения SileroTTSApp")
        
        try:
            # Инициализация pygame mixer (channels=2 для стерео)
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
            logging.info(f"Pygame mixer инициализирован: частота={SAMPLE_RATE}Hz, каналы=2")
        except Exception as e:
            logging.error(f"Ошибка инициализации pygame mixer: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось инициализировать звуковую систему: {e}")
            sys.exit(1)
        
        self.setup_ui()
        self.load_model_threaded()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        logging.info("Настройка пользовательского интерфейса")
        
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Статус загрузки модели
        self.status_var = tk.StringVar(value="Статус: Проверка модели...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Метка для текста
        ttk.Label(main_frame, text="Введите текст для озвучки:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        
        # Текстовое поле
        self.text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10, font=("Arial", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        default_text = "Это демонстрация работы Silero TTS версии 5. Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности! В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров."
        self.text_area.insert(tk.END, default_text)
        logging.debug(f"Текстовое поле инициализировано с текстом по умолчанию длиной {len(default_text)} символов")
        
        # Фрейм для выбора голоса и кнопок
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Выбор голоса
        ttk.Label(controls_frame, text="Выберите голос:").pack(side=tk.LEFT, padx=(0, 10))
        self.speaker_combo = ttk.Combobox(controls_frame, values=SPEAKERS, state="readonly", width=15)
        self.speaker_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.speaker_combo.current(0)  # baya по умолчанию
        logging.debug(f"Комбобокс голосов инициализирован со значениями {SPEAKERS}")
        
        # Кнопка воспроизведения
        self.play_btn = ttk.Button(controls_frame, text="▶ Сгенерировать и воспроизвести", command=self.play_audio_threaded)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка сохранения
        self.save_btn = ttk.Button(controls_frame, text="💾 Сохранить в WAV", command=self.save_audio_threaded)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка остановки
        self.stop_btn = ttk.Button(controls_frame, text="⏹ Остановить", command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка очистки лога
        self.clear_log_btn = ttk.Button(controls_frame, text="📋 Очистить лог", command=self.clear_log)
        self.clear_log_btn.pack(side=tk.LEFT, padx=5)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        logging.info("Пользовательский интерфейс настроен успешно")
    
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
            
            # Проверка наличия файла модели
            if not os.path.isfile(MODEL_FILE):
                logging.info(f"Файл модели {MODEL_FILE} не найден. Начинается скачивание...")
                self.update_status("Статус: Скачивание модели (первый запуск может занять время)...")
                
                # Проверка доступности URL
                try:
                    torch.hub.download_url_to_file(MODEL_URL, MODEL_FILE, progress=True)
                    logging.info(f"Модель успешно скачана: {MODEL_FILE}")
                except Exception as e:
                    logging.error(f"Ошибка при скачивании модели: {e}", exc_info=True)
                    raise Exception(f"Не удалось скачать модель: {e}")
            else:
                file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)  # в МБ
                logging.info(f"Файл модели {MODEL_FILE} уже существует (размер: {file_size:.2f} МБ)")
            
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
    
    def generate_audio(self, text, speaker):
        """Генерация аудио из текста"""
        if not self.is_model_loaded:
            raise Exception("Модель не загружена")
        
        logging.info(f"Генерация аудио: текст='{text[:100]}...' (длина: {len(text)}), голос='{speaker}'")
        
        try:
            # Генерация аудио
            audio = self.model.apply_tts(
                text=text,
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
    
    def start_progress(self):
        """Запуск прогресс-бара"""
        try:
            self.root.after(0, self.progress.start)
        except Exception as e:
            logging.error(f"Ошибка при запуске прогресс-бара: {e}")
    
    def stop_progress(self):
        """Остановка прогресс-бара"""
        try:
            self.root.after(0, self.progress.stop)
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
            self.update_status(f"Статус: Генерация речи голосом '{speaker}'...")
            self.start_progress()
            
            # Генерация аудио
            audio = self.generate_audio(text, speaker)
            logging.info("Аудио успешно сгенерировано")
            
            self.update_status("Статус: Воспроизведение...")
            
            # Конвертация тензора в numpy массив
            audio_numpy = audio.numpy()
            logging.debug(f"Размерность аудио: {audio_numpy.shape}")
            logging.debug(f"Диапазон значений: [{audio_numpy.min():.4f}, {audio_numpy.max():.4f}]")
            
            # Убедимся, что аудио в нужном диапазоне и формате
            if audio_numpy.max() > 1.0 or audio_numpy.min() < -1.0:
                max_val = np.abs(audio_numpy).max()
                if max_val > 0:
                    audio_numpy = audio_numpy / max_val
                    logging.info(f"Аудио нормализовано (макс. значение было {max_val:.4f})")
                else:
                    logging.warning("Аудио содержит только нули")
            
            # Преобразование в 16-битный формат
            audio_int16 = (audio_numpy * 32767).astype(np.int16)
            logging.debug(f"Аудио преобразовано в 16-битный формат, диапазон: [{audio_int16.min()}, {audio_int16.max()}]")
            
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
            
        except Exception as e:
            logging.error(f"Ошибка при воспроизведении аудио: {str(e)}", exc_info=True)
            self.update_status("Статус: Ошибка ❌")
            self.show_error("Ошибка", str(e))
            self.stop_progress()
    
    def stop_audio(self):
        """Остановка воспроизведения"""
        try:
            if pygame.mixer.get_init():  # Проверяем, инициализирован ли микшер
                pygame.mixer.stop()
                if self.current_sound:
                    self.current_sound.stop()
                    self.current_sound = None
                self.update_status("Статус: Воспроизведение остановлено")
                logging.info("Воспроизведение остановлено пользователем")
        except Exception as e:
            logging.error(f"Ошибка при остановке воспроизведения: {e}")
    
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
            self.update_status(f"Статус: Генерация для сохранения...")
            self.start_progress()
            
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"silero_{speaker}_{timestamp}.wav"
            
            logging.info(f"Сохранение в файл: {filename}")
            
            # Сохраняем файл
            self.model.save_wav(
                text=text,
                speaker=speaker,
                sample_rate=SAMPLE_RATE,
                audio_path=filename
            )
            
            # Проверяем, что файл создан
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / 1024  # в КБ
                logging.info(f"Файл {filename} успешно создан (размер: {file_size:.2f} КБ)")
                self.update_status(f"Статус: Сохранено в {filename} ✅")
                
                full_path = os.path.abspath(filename)
                self.show_info("Успех", f"Аудио сохранено в файл:\n{full_path}\n\nРазмер: {file_size:.2f} КБ")
            else:
                logging.error(f"Файл {filename} не был создан")
                raise Exception("Файл не был создан")
            
            self.stop_progress()
            
        except Exception as e:
            logging.error(f"Ошибка при сохранении аудио: {str(e)}", exc_info=True)
            self.update_status("Статус: Ошибка сохранения ❌")
            self.show_error("Ошибка", str(e))
            self.stop_progress()
    
    def clear_log(self):
        """Очистка лог-файла"""
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"Лог очищен {datetime.now()}\n")
            logging.info("Лог-файл очищен")
            self.show_info("Лог", f"Лог-файл очищен:\n{log_filename}")
        except Exception as e:
            logging.error(f"Ошибка при очистке лога: {e}")
            self.show_error("Ошибка", f"Не удалось очистить лог: {e}")
    
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
        self.cleanup()
        self.root.destroy()
        logging.shutdown()

def main():
    """Главная функция запуска приложения"""
    logging.info("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
    print(f"Лог-файл: {log_filename}")
    
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

if __name__ == "__main__":
    main()