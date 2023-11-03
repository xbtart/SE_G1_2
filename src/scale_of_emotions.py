from transformers import pipeline

# Пайплайн для машинного перевода
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Пайплайн для анализа эмоциональной окраски текста
emotion_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Запрос ввода текста с клавиатуры
input_text = input("Введите текст на английском: ")

# Выполнение машинного перевода
translated_text = translator(input_text)

# Оценка эмоциональной окраски переведенного текста
emotion_result = emotion_analyzer(translated_text[0]['translation_text'])

# Вывод эмоциональной оценки
print("Шкала эмоций:", round(emotion_result[0]['score'] * 100), "%")
if emotion_result[0]['label'] == '5 stars' and emotion_result[0]['score'] > 0.5:
    print("Поздравляем! Ваша фраза удачная, вот перевод на французский:")
    print(translated_text[0]['translation_text'])
else:
    print("Опс! Мы не переводим фразы с низкой оценкой по шкале эмоций. Попробуйте быть более открытыми и подельться своими чувствами...")