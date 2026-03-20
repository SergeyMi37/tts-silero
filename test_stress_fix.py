#!/usr/bin/env python3
"""
Тест для проверки исправления метода apply_stress_marks
для обработки длинных текстов.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_preprocessor import TextPreprocessor
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_short_text():
    """Тест с коротким текстом."""
    print("\n=== Тест с коротким текстом ===")
    
    preprocessor = TextPreprocessor()
    short_text = "Привет, мир! Это короткий текст для проверки."
    
    try:
        result = preprocessor.apply_stress_marks(short_text)
        print(f"Исходный текст: {short_text}")
        print(f"Результат: {result}")
        print(f"Длина исходного: {len(short_text)}, результата: {len(result)}")
        return True
    except Exception as e:
        print(f"Ошибка при обработке короткого текста: {e}")
        return False

def test_long_text():
    """Тест с длинным текстом."""
    print("\n=== Тест с длинным текстом ===")
    
    preprocessor = TextPreprocessor()
    
    # Создаем длинный текст (больше 1500 символов)
    long_text = """
    Это очень длинный текст для проверки работы исправленного метода apply_stress_marks.
    Метод должен разбить этот текст на части по примерно 1500 символов каждая, сохраняя при этом границы предложений.
    
    Первое предложение содержит много слов и должно быть обработано корректно. Второе предложение также важно для проверки.
    Третье предложение добавляет еще больше контента. Четвертое предложение продолжает увеличивать размер текста.
    
    Пятое предложение содержит различные слова для проверки расстановки ударений. Шестое предложение также важно.
    Седьмое предложение добавляет разнообразия в текст. Восьмое предложение продолжает тестирование.
    
    Девятое предложение проверяет работу с большими объемами текста. Десятое предложение завершает первую часть.
    Одиннадцатое предложение начинает новую секцию текста. Двенадцатое предложение продолжает проверку.
    
    Тринадцатое предложение содержит сложные слова для обработки. Четырнадцатое предложение также важно для тестирования.
    Пятнадцатое предложение добавляет еще больше контента для проверки. Шестнадцатое предложение продолжает тест.
    
    Семнадцатое предложение проверяет стабильность работы алгоритма. Восемнадцатое предложение также критично.
    Девятнадцатое предложение добавляет финальный контент. Двадцатое предложение завершает длинный тест.
    
    Дополнительный текст для увеличения общего размера и проверки разбиения на части. Еще одно предложение для тестирования.
    Последнее предложение в этом длинном тексте должно быть обработано корректно и без ошибок.
    """.strip()
    
    print(f"Длина исходного текста: {len(long_text)} символов")
    
    try:
        result = preprocessor.apply_stress_marks(long_text)
        print(f"Длина результата: {len(result)} символов")
        print(f"Первые 200 символов результата: {result[:200]}...")
        print(f"Последние 200 символов результата: ...{result[-200:]}")
        return True
    except Exception as e:
        print(f"Ошибка при обработке длинного текста: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunk_splitting():
    """Тест разбиения текста на части."""
    print("\n=== Тест разбиения текста на части ===")
    
    preprocessor = TextPreprocessor()
    
    test_text = "Первое предложение. Второе предложение! Третье предложение? Четвертое предложение."
    chunks = preprocessor._split_text_into_chunks(test_text, max_chunk_size=50)
    
    print(f"Исходный текст: {test_text}")
    print(f"Разбито на {len(chunks)} частей:")
    for i, chunk in enumerate(chunks):
        print(f"  Часть {i+1} ({len(chunk)} символов): '{chunk}'")
    
    return True

def main():
    """Основная функция тестирования."""
    print("Тестирование исправленного метода apply_stress_marks")
    print("=" * 60)
    
    tests = [
        ("Разбиение на части", test_chunk_splitting),
        ("Короткий текст", test_short_text),
        ("Длинный текст", test_long_text),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\nЗапуск теста: {test_name}")
            success = test_func()
            results.append((test_name, success))
            print(f"Тест '{test_name}': {'ПРОЙДЕН' if success else 'ПРОВАЛЕН'}")
        except Exception as e:
            print(f"Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    for test_name, success in results:
        status = "✓ ПРОЙДЕН" if success else "✗ ПРОВАЛЕН"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    print(f"\nОбщий результат: {'ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_passed else 'ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
