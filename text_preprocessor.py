"""
Модуль предобработки текста для TTS-приложения Silero.

Обеспечивает:
- Преобразование чисел в слова (num2words)
- Расстановку ударений (silero-stress + ruaccent опционально)
- Обработку сокращений (ул. → улица, г. → город)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Класс для предобработки текста перед синтезом речи.
    
    Поддерживает ленивую инициализацию моделей для ускорения старта приложения.
    Формат ударений: +перед гласной (совместимо с Silero TTS v5).
    """
    
    # Словарь распространённых сокращений
    ABBREVIATIONS = {
        'ул.': 'улица',
        'ул': 'улица',
        'пер.': 'переулок',
        'пер': 'переулок',
        'пр.': 'проспект',
        'пр': 'проспект',
        'пл.': 'площадь',
        'пл': 'площадь',
        'г.': 'город',
        'г': 'город',
        'д.': 'дом',
        'д': 'дом',
        'корп.': 'корпус',
        'корп': 'корпус',
        'стр.': 'строение',
        'стр': 'строение',
        'кв.': 'квартира',
        'кв': 'квартира',
        'эт.': 'этаж',
        'эт': 'этаж',
        'под.': 'подъезд',
        'под': 'подъезд',
        'р-н': 'район',
        'р-н.': 'район',
        'обл.': 'область',
        'обл': 'область',
        'край': 'край',
        'респ.': 'республика',
        'респ': 'республика',
        'авто': 'автомобиль',
        'т.': 'товарищ',
        'т': 'товарищ',
        'г-н': 'господин',
        'г-н.': 'господин',
        'г-жа': 'госпожа',
        'г-жа.': 'госпожа',
        'мл.': 'младший',
        'мл': 'младший',
        'ст.': 'старший',
        'ст': 'старший',
        'им.': 'имени',
        'им': 'имени',
        'акад.': 'академик',
        'акад': 'академик',
        'проф.': 'профессор',
        'проф': 'профессор',
        'доц.': 'доцент',
        'доц': 'доцент',
        'докт.': 'доктор',
        'докт': 'доктор',
        'канд.': 'кандидат',
        'канд': 'кандидат',
        'тыс.': 'тысяча',
        'тыс': 'тысяча',
        'млн': 'миллион',
        'млн.': 'миллион',
        'млрд': 'миллиард',
        'млрд.': 'миллиард',
        'руб.': 'рубль',
        'руб': 'рубль',
        'коп.': 'копейка',
        'коп': 'копейка',
        'кг.': 'килограмм',
        'кг': 'килограмм',
        'г.': 'грамм',
        'г': 'грамм',
        'м.': 'метр',
        'м': 'метр',
        'см.': 'сантиметр',
        'см': 'сантиметр',
        'мм.': 'миллиметр',
        'мм': 'миллиметр',
        'км.': 'километр',
        'км': 'километр',
        'сек.': 'секунда',
        'сек': 'секунда',
        'мин.': 'минута',
        'мин': 'минута',
        'ч.': 'час',
        'ч': 'час',
        'т.е.': 'то есть',
        'т. д.': 'так далее',
        'т. п.': 'тому подобное',
        'и т.д.': 'и так далее',
        'и т.п.': 'и тому подобное',
        'напр.': 'например',
        'напр': 'например',
        'см.': 'смотри',
        'см': 'смотри',
        'ср.': 'сравни',
        'ср': 'сравни',
    }
    
    def __init__(self):
        """Инициализация препроцессора без загрузки моделей."""
        self._stress_model = None
        self._ruaccent_model = None
        self._models_loaded = False
        logger.info("TextPreprocessor инициализирован (модели не загружены)")
    
    @property
    def is_loaded(self) -> bool:
        """Проверка загруженности моделей."""
        return self._models_loaded
    
    def load_models(self, use_ruaccent: bool = False) -> None:
        """
        Ленивая загрузка моделей для расстановки ударений.
        
        Args:
            use_ruaccent: Если True, загружает ruaccent для дополнительной обработки
        """
        if self._models_loaded:
            logger.info("Модели уже загружены")
            return
        
        try:
            logger.info("Начало загрузки моделей предобработки текста")
            
            # Загрузка silero-stress
            logger.info("Загрузка silero-stress модели...")
            from silero_stress import load_accentor
            self._stress_model = load_accentor()
            logger.info("silero-stress модель загружена успешно")
            
            # Опциональная загрузка ruaccent
            if use_ruaccent:
                try:
                    logger.info("Загрузка ruaccent модели...")
                    import ruaccent
                    self._ruaccent_model = ruaccent
                    logger.info("ruaccent модель загружена успешно")
                except ImportError as e:
                    logger.warning(f"ruaccent не установлен: {e}. Будет использоваться только silero-stress.")
                    self._ruaccent_model = None
                    # Не прерываем загрузку, silero-stress продолжит работать
            else:
                logger.info("ruaccent модель не загружена (use_ruaccent=False)")
                self._ruaccent_model = None
            
            self._models_loaded = True
            logger.info("Все модели предобработки успешно загружены")
            
        except ImportError as e:
            logger.error(f"Ошибка импорта моделей: {e}. Убедитесь, что зависимости установлены.")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {e}", exc_info=True)
            raise
    
    def preprocess(
        self, 
        text: str, 
        use_num2words: bool = True, 
        use_ruaccent: bool = False
    ) -> str:
        """
        Полная предобработка текста.
        
        Args:
            text: Исходный текст для обработки
            use_num2words: Если True, преобразует числа в слова
            use_ruaccent: Если True, использует ruaccent для дополнительной обработки
        
        Returns:
            Обработанный текст с расставленными ударениями
        """
        if not text or not text.strip():
            return text
        
        logger.info(f"Начало предобработки текста (длина: {len(text)})")
        
        result = text
        
        # Шаг 1: Преобразование чисел в слова
        if use_num2words:
            result = self.replace_numbers_with_words(result)
        
        # Шаг 2: Обработка сокращений
        result = self.process_abbreviations(result)
        
        # Шаг 3: Расстановка ударений
        result = self.apply_stress_marks(result, use_ruaccent=use_ruaccent)
        
        logger.info(f"Предобработка завершена (длина результата: {len(result)})")
        return result
    
    def replace_numbers_with_words(self, text: str) -> str:
        """
        Преобразование чисел в слова с помощью num2words.
        
        Поддерживает:
        - Целые числа (123 → сто двадцать три)
        - Четырехзначные числа (1955 → одна тысяча пятьдесят пять)
        - Четырехзначные числа перед "год" (1955 год → одна тысяча пятьдесят пятый год)
        - Десятичные числа (3.14 → три целых четырнадцать сотых)
        - Числа с разделителями тысяч (1 000 000 → один миллион)
        
        Args:
            text: Текст с числами
        
        Returns:
            Текст с числами, преобразованными в слова
        """
        if not self._models_loaded:
            # Загружаем только num2words по требованию (не требует ML-моделей)
            pass
        
        try:
            from num2words import num2words
        except ImportError:
            logger.warning("num2words не установлен, преобразование чисел пропущено")
            return text
        
        def get_ordinal_form(cardinal_text: str) -> str:
            """
            Преобразует кардинальное число в порядственное.
            Например: "пятьдесят пять" → "пятьдесят пятый"
            """
            # Словарь для преобразования окончаний
            ordinal_mappings = {
                'один': 'первый',
                'два': 'второй',
                'три': 'третий',
                'четыре': 'четвертый',
                'пять': 'пятый',
                'шесть': 'шестой',
                'семь': 'седьмой',
                'восемь': 'восьмой',
                'девять': 'девятый',
                'десять': 'десятый',
                'одиннадцать': 'одиннадцатый',
                'двенадцать': 'двенадцатый',
                'тринадцать': 'тринадцатый',
                'четырнадцать': 'четырнадцатый',
                'пятнадцать': 'пятнадцатый',
                'шестнадцать': 'шестнадцатый',
                'семнадцать': 'семнадцатый',
                'восемнадцать': 'восемнадцатый',
                'девятнадцать': 'девятнадцатый',
                'двадцать': 'двадцатый',
                'тридцать': 'тридцатый',
                'сорок': 'сороковой',
                'пятьдесят': 'пятидесятый',
                'шестьдесят': 'шестидесятый',
                'семьдесят': 'семидесятый',
                'восемьдесят': 'восьмидесятый',
                'девяносто': 'девяностый',
                'сто': 'сотый',
                'двести': 'двухсотый',
                'триста': 'трехсотый',
                'четыреста': 'четырехсотый',
                'пятьсот': 'пятисотый',
                'шестьсот': 'шестисотый',
                'семьсот': 'семисотый',
                'восемьсот': 'восьмисотый',
                'девятьсот': 'девяти сотый',
            }
            
            words = cardinal_text.split()
            if not words:
                return cardinal_text
            
            # Изменяем только последнее слово
            last_word = words[-1].lower()
            if last_word in ordinal_mappings:
                words[-1] = ordinal_mappings[last_word]
            
            return ' '.join(words)
        
        def replace_number(match, check_year=False):
            num_str = match.group(0)
            next_word = match.group('next_word') if check_year else None
            
            try:
                # Очистка от разделителей тысяч (пробелы, запятые)
                clean_num = num_str.replace(' ', '').replace(',', '.')
                
                # Проверка на четырехзначное число (год)
                is_four_digit = re.match(r'^\d{4}$', clean_num) is not None
                
                # Проверка на десятичное число
                if '.' in clean_num:
                    parts = clean_num.split('.')
                    if len(parts) == 2 and parts[0] and parts[1]:
                        integer_part = int(parts[0]) if parts[0] else 0
                        decimal_part = parts[1]
                        
                        # Преобразование целой части
                        if integer_part == 0:
                            result = num2words(0, lang='ru')
                        else:
                            result = num2words(integer_part, lang='ru')
                        
                        # Добавление дробной части
                        if integer_part != 0:
                            result += ' целых '
                        
                        # Склонение дробной части
                        last_digit = decimal_part[-1] if decimal_part else '0'
                        if len(decimal_part) == 1:
                            if last_digit == '1':
                                result += num2words(int(decimal_part), lang='ru') + ' десятая'
                            else:
                                result += num2words(int(decimal_part), lang='ru') + ' десятых'
                        elif len(decimal_part) == 2:
                            if decimal_part == '10':
                                result += num2words(10, lang='ru') + ' сотых'
                            elif 11 <= int(decimal_part) <= 19:
                                result += num2words(int(decimal_part), lang='ru') + ' сотых'
                            else:
                                result += num2words(int(decimal_part), lang='ru') + ' сотых'
                        elif len(decimal_part) == 3:
                            result += num2words(int(decimal_part), lang='ru') + ' тысячных'
                        else:
                            # Для длинных дробей просто читаем цифры
                            result += ' '.join(num2words(int(d), lang='ru') for d in decimal_part)
                        
                        return result
                    else:
                        # Неверный формат, возвращаем как есть
                        return num_str
                else:
                    # Целое число
                    num = int(clean_num)
                    
                    # Специальная обработка четырехзначных чисел (годы)
                    if is_four_digit and 1000 <= num <= 9999:
                        # Разбиваем на тысячи и остаток
                        thousands = num // 1000
                        remainder = num % 1000
                        
                        # Получаем слово для тысяч
                        if thousands == 1:
                            result = 'одна тысяча'
                        else:
                            result = num2words(thousands, lang='ru') + ' тысячи'
                        
                        # Добавляем остаток
                        if remainder > 0:
                            remainder_words = num2words(remainder, lang='ru')
                            result += ' ' + remainder_words
                        
                        # Если за числом следует "год", используем порядковое окончание
                        if check_year and next_word and next_word.lower() in ['год', 'года', 'году', 'годом', 'годе']:
                            result = get_ordinal_form(result)
                        
                        return result
                    else:
                        # Обычное число
                        return num2words(num, lang='ru')
                    
            except (ValueError, OverflowError) as e:
                logger.warning(f"Не удалось преобразовать число {num_str}: {e}")
                return num_str
            except Exception as e:
                logger.warning(f"Ошибка при преобразовании числа {num_str}: {e}")
                return num_str
        
        # Паттерн для поиска чисел с проверкой на "год" после
        # Поддерживает: 123, 1 000, 1.5, 3,14, 1 000 000, 1955 год
        # Сначала обрабатываем 4-значные числа перед "год"
        number_pattern_with_year = r'(?<![а-яА-ЯёЁa-zA-Z0-9])(\d{4})\s+(год|года|году|годом|годе)(?![а-яА-ЯёЁa-zA-Z0-9])'
        number_pattern = r'(?<![а-яА-ЯёЁa-zA-Z0-9])\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?(?![а-яА-ЯёЁa-zA-Z0-9])'
        
        # Сначала обрабатываем числа с "год"
        def replace_with_year(match):
            num_str = match.group(1)
            year_word = match.group(2)
            
            # Создаем фейковый match объект для replace_number
            class FakeMatch:
                def group(self, n=0):
                    if n == 0:
                        return num_str
                    return num_str
                @property
                def string(self):
                    return text
            
            fake_match = FakeMatch()
            cardinal_result = replace_number(fake_match, check_year=False)
            ordinal_result = get_ordinal_form(cardinal_result)
            return ordinal_result + ' ' + year_word
        
        result = re.sub(number_pattern_with_year, replace_with_year, text)
        
        # Затем обрабатываем остальные числа
        result = re.sub(number_pattern, lambda m: replace_number(m, check_year=False), result)
        
        logger.info(f"Числа преобразованы в слова: '{text[:50]}...' → '{result[:50]}...'")
        return result
    
    def apply_stress_marks(self, text: str, use_ruaccent: bool = False) -> str:
        """
        Расстановка ударений в тексте.
        
        Использует silero-stress для расстановки ударений.
        Опционально использует ruaccent для дополнительной обработки.
        Для длинных текстов разбивает их на части по ~1500 символов
        с сохранением границ предложений.
        
        Формат: +перед гласной (совместимо с Silero TTS v5)
        Пример: "замок" → "з+амок" или "зам+ок"
        
        Args:
            text: Текст для расстановки ударений
            use_ruaccent: Если True, использует ruaccent для дополнительной обработки
        
        Returns:
            Текст с расставленными ударениями
        """
        if not text.strip():
            return text
        
        # Загрузка моделей если нужно
        if not self._models_loaded:
            self.load_models(use_ruaccent=use_ruaccent)
        
        try:
            # Применение silero-stress
            if self._stress_model:
                logger.debug("Применение silero-stress для расстановки ударений")
                
                # Если текст короткий, обрабатываем сразу
                if len(text) <= 1500:
                    logger.debug(f"Применение silero-stress к тексту (длина: {len(text)})")
                    result = self._stress_model(text)
                    logger.debug(f"Результат silero-stress (первые 200 симв.): {result[:200]}")
                else:
                    # Разбиваем длинный текст на части
                    logger.info(f"Текст длинный ({len(text)} символов), разбиваем на части")
                    chunks = self._split_text_into_chunks(text, max_chunk_size=1500)
                    processed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Обработка части {i+1}/{len(chunks)} (длина: {len(chunk)})")
                        try:
                            processed_chunk = self._stress_model(chunk)
                            logger.debug(f"Часть {i+1} результат (первые 100 симв.): {processed_chunk[:100]}")
                            processed_chunks.append(processed_chunk)
                        except Exception as chunk_error:
                            logger.warning(f"Ошибка при обработке части {i+1}: {chunk_error}")
                            # При ошибке добавляем исходный текст части
                            processed_chunks.append(chunk)
                    
                    result = ''.join(processed_chunks)
                    logger.info(f"Обработка частей завершена, итоговая длина: {len(result)}")
                    logger.debug(f"Результат silero-stress (первые 200 симв.): {result[:200]}")
            else:
                result = text
            
            # Опциональное применение ruaccent
            if use_ruaccent and self._ruaccent_model:
                try:
                    logger.debug("Применение ruaccent для дополнительной обработки")
                    # ruaccent может улучшить качество расстановки ударений
                    # Проверяем доступный API
                    if hasattr(self._ruaccent_model, 'process_all'):
                        result = self._ruaccent_model.process_all(result)
                    elif hasattr(self._ruaccent_model, 'accent_line'):
                        # Для ru-accent-poet
                        result = self._ruaccent_model.accent_line(result)
                    else:
                        logger.warning("ruaccent: неизвестный API, пропускаем дополнительную обработку")
                except Exception as ruaccent_error:
                    logger.warning(f"ruaccent не применил ударения: {ruaccent_error}. Используем только silero-stress.")
            
            logger.info(f"Ударения расставлены (длина: {len(result)})")
            
        except Exception as e:
            logger.error(f"Ошибка при расстановке ударений: {e}", exc_info=True)
            # Возвращаем исходный текст при ошибке
            result = text
        
        return result
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1500) -> list[str]:
        """
        Разбивает текст на части с сохранением границ предложений.
        
        Args:
            text: Исходный текст для разбиения
            max_chunk_size: Максимальный размер части в символах
        
        Returns:
            Список частей текста
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Разбиваем текст на предложения по знакам препинания
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(f'({sentence_endings})', text)
        
        # Объединяем предложения с их знаками препинания
        processed_sentences = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences) and re.match(sentence_endings, sentences[i + 1]):
                # Объединяем предложение с его окончанием
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1
            
            if sentence.strip():  # Пропускаем пустые строки
                processed_sentences.append(sentence)
        
        # Если не удалось разбить на предложения, разбиваем по абзацам
        if len(processed_sentences) <= 1:
            processed_sentences = text.split('\n')
        
        # Если и абзацев мало, разбиваем по словам
        if len(processed_sentences) <= 1:
            words = text.split()
            processed_sentences = []
            current_sentence = ""
            
            for word in words:
                if len(current_sentence + " " + word) <= max_chunk_size // 3:
                    current_sentence += (" " if current_sentence else "") + word
                else:
                    if current_sentence:
                        processed_sentences.append(current_sentence)
                    current_sentence = word
            
            if current_sentence:
                processed_sentences.append(current_sentence)
        
        # Собираем части, не превышающие max_chunk_size
        for sentence in processed_sentences:
            # Если одно предложение больше лимита, разбиваем его принудительно
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Разбиваем длинное предложение на части
                while len(sentence) > max_chunk_size:
                    # Ищем последний пробел в пределах лимита
                    split_pos = sentence.rfind(' ', 0, max_chunk_size)
                    if split_pos == -1:
                        split_pos = max_chunk_size
                    
                    chunks.append(sentence[:split_pos])
                    sentence = sentence[split_pos:].lstrip()
                
                if sentence:
                    current_chunk = sentence
            else:
                # Проверяем, поместится ли предложение в текущую часть
                if len(current_chunk + sentence) <= max_chunk_size:
                    current_chunk += sentence
                else:
                    # Сохраняем текущую часть и начинаем новую
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        # Добавляем последнюю часть
        if current_chunk:
            chunks.append(current_chunk)
        
        # Если получилась только одна часть, но она слишком большая,
        # разбиваем принудительно
        if len(chunks) == 1 and len(chunks[0]) > max_chunk_size:
            text_to_split = chunks[0]
            chunks = []
            
            while len(text_to_split) > max_chunk_size:
                split_pos = text_to_split.rfind(' ', 0, max_chunk_size)
                if split_pos == -1:
                    split_pos = max_chunk_size
                
                chunks.append(text_to_split[:split_pos])
                text_to_split = text_to_split[split_pos:].lstrip()
            
            if text_to_split:
                chunks.append(text_to_split)
        
        logger.debug(f"Текст разбит на {len(chunks)} частей: {[len(chunk) for chunk in chunks]}")
        return chunks
    
    def process_abbreviations(self, text: str) -> str:
        """
        Обработка сокращений и аббревиатур.
        
        Заменяет распространённые сокращения на полные формы:
        - ул. → улица
        - г. → город
        - д. → дом
        - и т.д. → и так далее
        
        Args:
            text: Текст с сокращениями
        
        Returns:
            Текст с раскрытыми сокращениями
        """
        if not text.strip():
            return text
        
        logger.debug("Начало обработки сокращений")
        result = text
        
        # Сортируем ключи по длине (убывание) для корректной замены
        # Сначала заменяем более длинные сокращения (и т.д. перед т.д.)
        sorted_abbrevs = sorted(self.ABBREVIATIONS.keys(), key=len, reverse=True)
        
        for abbrev in sorted_abbrevs:
            # Экранируем специальные символы для regex
            escaped = re.escape(abbrev)
            # Паттерн с границами слов и учетом регистра
            pattern = r'\b' + escaped + r'\b'
            
            def replace_func(match):
                original = match.group(0)
                replacement = self.ABBREVIATIONS[abbrev]
                
                # Сохраняем регистр первой буквы
                if original[0].isupper():
                    return replacement.capitalize()
                elif original.isupper():
                    return replacement.upper()
                else:
                    return replacement
            
            result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)
        
        logger.debug("Обработка сокращений завершена")
        return result
    
    def unload_models(self) -> None:
        """Выгрузка моделей из памяти для освобождения ресурсов."""
        self._stress_model = None
        self._ruaccent_model = None
        self._models_loaded = False
        logger.info("Модели предобработки выгружены из памяти")
