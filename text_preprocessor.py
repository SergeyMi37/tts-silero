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
                logger.info("Загрузка ruaccent модели...")
                import ruaccent
                self._ruaccent_model = ruaccent
                logger.info("ruaccent модель загружена успешно")
            else:
                logger.info("ruaccent модель не загружена (use_ruaccent=False)")
            
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
        
        def replace_number(match):
            num_str = match.group(0)
            try:
                # Очистка от разделителей тысяч (пробелы, запятые)
                clean_num = num_str.replace(' ', '').replace(',', '.')
                
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
                    return num2words(num, lang='ru')
                    
            except (ValueError, OverflowError) as e:
                logger.warning(f"Не удалось преобразовать число {num_str}: {e}")
                return num_str
            except Exception as e:
                logger.warning(f"Ошибка при преобразовании числа {num_str}: {e}")
                return num_str
        
        # Паттерн для поиска чисел (целые и десятичные, с разделителями тысяч)
        # Поддерживает: 123, 1 000, 1.5, 3,14, 1 000 000
        # Убраны \b границы для лучшей работы с русским текстом
        number_pattern = r'(?<![а-яА-ЯёЁa-zA-Z])\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?(?![а-яА-ЯёЁa-zA-Z])'
        
        result = re.sub(number_pattern, replace_number, text)
        logger.info(f"Числа преобразованы в слова: '{text[:50]}...' → '{result[:50]}...'")
        return result
    
    def apply_stress_marks(self, text: str, use_ruaccent: bool = False) -> str:
        """
        Расстановка ударений в тексте.
        
        Использует silero-stress для расстановки ударений.
        Опционально использует ruaccent для дополнительной обработки.
        
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
        
        result = text
        
        try:
            # Применение silero-stress
            if self._stress_model:
                logger.debug("Применение silero-stress для расстановки ударений")
                result = self._stress_model(text)
            
            # Опциональное применение ruaccent
            if use_ruaccent and self._ruaccent_model:
                logger.debug("Применение ruaccent для дополнительной обработки")
                # ruaccent может улучшить качество расстановки ударений
                result = self._ruaccent_model.process_all(result)
            
            logger.info(f"Ударения расставлены (длина: {len(result)})")
            
        except Exception as e:
            logger.error(f"Ошибка при расстановке ударений: {e}", exc_info=True)
            # Возвращаем исходный текст при ошибке
            result = text
        
        return result
    
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
