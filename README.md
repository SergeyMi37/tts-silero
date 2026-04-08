# Silero TTS - Озвучка текста

Приложение для преобразования текста в речь с использованием модели Silero TTS v5.

## Установка

```bash
# Клонировать проект
git clone <repo_url>
cd <repo_name>

# Создать виртуальное окружение
python -m venv env
env\Scripts\activate  # Windows
# source env/bin/activate  # Linux/Mac

# Установить зависимости
pip install -r requirements.txt
```

## Запуск

```bash
python text2mp3.py
```

## Использование (CLI)

```bash
# Озвучка текста
python text2mp3.py --text "Привет мир!" -o output.mp3

# Из файла
python text2mp3.py -i book.txt -o audiobook.mp3 --chunks

# Выбор голоса и скорости
python text2mp3.py -i book.txt -o output.mp3 --speaker eugene --speech-rate slow --chunks

# С предобработкой и ударениями
python text2mp3.py -i book.txt -o output.mp3 --chunks --preprocess --ruaccent
```

## Параметры CLI

| Параметр | Описание |
|----------|----------|
| `--text`, `-t` | Текст для озвучки |
| `--input-file`, `-i` | Файл (txt, fb2, zip) |
| `--output`, `-o` | Выходной файл MP3 |
| `--speaker`, `-s` | Голос: baya, eugene, kseniya, xenia, random |
| `--speech-rate`, `-r` | Скорость: x-slow, slow, medium, fast, x-fast |
| `--chunks` | Разбить на чанки |
| `--max-chars` | Макс. символов в чанке (по умолчанию 1200) |
| `--silence-ms` | Пауза между чанками в мс |
| `--preprocess` | Предобработка текста |
| `--ruaccent` | Расстановка ударений через ruaccent |
| `--save-parts` | Сохранить чанки в отдельную папку |
| `--delete-parts` | Удалить папку с чанками после завершения |
| `--output-dir` | Папка для сохранения |
| `--bitrate` | Битрейт MP3 (128k, 192k, 256k, 320k) |

## Требования

- Python 3.8+
- torch
- silero-tts-enhanced
- pygame, pydub
- num2words, silero-stress, ruaccent
- **ffmpeg** (для объединения MP3) — подробнее ниже

## Установка ffmpeg

### Windows

**Вариант 1: Через winget (рекомендуется)**
```bash
winget install ffmpeg
```

**Вариант 2: Скачать бинарники вручную**
1. Скачайте ffmpeg с https://ffmpeg.org/download.html#build-windows
2. Выберите `essentials` или `full` архив (например, `ffmpeg-master-latest-win64-gpl.zip`)
3. Распакуйте архив (например, в `C:\ffmpeg`)
4. Добавьте путь в переменную PATH:
   ```powershell
   # В PowerShell (от администратора)
   $env:PATH += ";C:\ffmpeg\bin"
   ```
   Или через GUI: Параметры → Система → О системе → Дополнительные параметры → Переменные среды → PATH → Изменить

**Вариант 3: Через Chocolatey**
```bash
choco install ffmpeg
```

**Вариант 4: Через Scoop**
```bash
scoop install ffmpeg
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### Linux (Fedora)

```bash
sudo dnf install ffmpeg
```

### Linux (Arch)

```bash
sudo pacman -S ffmpeg
```

### macOS

**Вариант 1: Через Homebrew (рекомендуется)**
```bash
brew install ffmpeg
```

**Вариант 2: MacPorts**
```bash
sudo port install ffmpeg
```

### Проверка установки

```bash
ffmpeg -version
```

Должно вывести информацию о версии ffmpeg. Если команда не найдена — перезапустите терминал или перезагрузите компьютер.

## Компиляция в EXE

### Установка PyInstaller

```bash
pip install pyinstaller
```

### Создание EXE файла

Используйте готовый spec-файл, который уже содержит все необходимые настройки:

```bash
pyinstaller text2mp3.spec
```

Готовый файл появится в папке `dist/text2mp3.exe`.

### Ручная сборка (альтернатива)

```bash
pyinstaller --onefile --windowed --name text2mp3 --icon=book-dyn.ico --add-data "book-dyn.ico;." --hidden-import=silero_stress.data --hidden-import=silero_stress --hidden-import=num2words --hidden-import=num2words.lang_RU text2mp3.py
```

| Параметр | Описание |
|----------|----------|
| `--onefile` | Упаковать всё в один exe файл |
| `--windowed` | Запуск без консольного окна |
| `--name` | Имя выходного файла |
| `--icon` | Иконка exe файла |
| `--add-data "файл;."` | Встроить файл данных в exe (Windows) |
| `--hidden-import` | Добавить скрытый импорт (нужен для silero_stress.data) |

### Примечания

- При первом запуске может потребоваться скачать модели Silero TTS
- Файл будет весить ~200-400 МБ из-за включения Python и зависимостей
- ffmpeg должен быть установлен в системе отдельно

## Лицензия

Apache 2.0 (модель Silero TTS)
