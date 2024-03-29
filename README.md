# SE_G1_2
Группа 1.2 по предмету Программная инженерия

# Миронов Артур Викторович  src/bad_topic.py
Этот скрипт предназначен для обработки текста на русском языке с целью выявления чувствительных тем. Если в тексте присутствуют чувствительные темы, скрипт выведет их на экран. В случае, если чувствительные темы отсутствуют, пользователь получит соответствующее уведомление.

Перед использованием необходимо установить библиотеку Hugging Face с помощью команды !pip install transformers.

Важно: Данный скрипт работает с корректным русскоязычным текстом. Для проверки корректности текста следует использовать другие инструменты или пакеты.

Примеры работы скрипта:

    Пользователь вводит текст: "Здесь нет никаких чувствительных тем." Результат: "Плохая тема не затронута."

    Пользователь вводит текст: "Текст с упоминанием острой темы." Результат: "Распознаны следующие острые темы: [название темы]!"

    Пользователь вводит текст на другом языке: "Text in another language." Результат: "Плохая тема не затронута."

Примечание: для определения чувствительных тем используется модель apanc/russian-sensitive-topics, предварительно обученная на соответствующем датасете.

#Драчев Алексей Анатольевич aleksrf@gmail.com
Добавлен файл src/scale_of_emotions.py
---------------------------------------
Данный файл предназначен для ввода текста на английском языке и 
последующий перевод данного текста на французкий, при условии, что оценка по "шкале эмоций" выше чем 50%
При недостаточной оценки перевод не осуществляется

Для работы данного файла требуется установить библиотеку Hugging Face
!pip install transformers sentencepiece

Замечание: Данный файл принимает правильный текст на английском языке, для определения правильности текста используйте другие пакеты

Примеры:
1) Введите текст на английском: person today
Шкала эмоций: 39 %
Опс! Мы не переводим фразы с низкой оценкой по шкале эмоций. Попробуйте быть более открытыми и подельться своими чувствами...

2) Введите текст на английском: I am the happiest person today
Шкала эмоций: 89 %
Поздравляем! Ваша фраза удачная, вот перевод на французский:
Je suis la personne la plus heureuse aujourd'hui.

3) Введите текст на английском: текст не на английском
Шкала эмоций: 43 %
Опс! Мы не переводим фразы с низкой оценкой по шкале эмоций. Попробуйте быть более открытыми и подельться своими чувствами...

# Баканов Максим Олегович - модель tiny2/cointegratedrubert_tiny2_cedr_emotion_detection.py


Представленная модель `cointegrated/rubert-tinyy2` для определения эмоциональной окраски текста на русском языке. Модель использует `RuBERT`, русскую версию модели `BERT`, и была обучена на датасете с разметкой эмоциональной окраски русскоязычных текстов. Модель способна определить эмоцию, содержащуюся в предложении или фразе, и классифицирует её на одну из следующих категорий: "sadness" - грусть/печаль, "anger" - гнев/ярость, "surprise" - удивление/сюрприз, "no_emotion" - отсутствие эмоций/безразличие, "fear" - страх/боязнь, "joy" - радость/веселье.


## Установка и использование

1. Установка зависимостей:
   ```
   pip install transformers
   ```

2. Импорт модуля pipeline из библиотеки transformers:
   ```python
   from transformers import pipeline
   ```

3. Загрузка модели:
   ```python
   classifier = pipeline("text-classification", "cointegrated/rubert-tiny2-cedr-emotion-detection")
   ```
   
4. Классификация текста:
   ```python
   text = "Добавьте муку и перемешайте - вы сами удивитесь, насколько все быстро произойдет!"
   results = classifier(text)
   print(results)
   ```
   
   В результате выполнения этого кода будет выведен словарь с ключами "label" и "score". "label" содержит обозначение класса эмоции ("sadness", "anger", "surprise", "no_emotion", "fear", "joy"), а "score" представляет собой численное значение, отражающее уверенность модели в принадлежности текста к определенному классу.

## Примеры использования

Пример использования модели:
```python
from transformers import pipeline

classifier = pipeline("text-classification", "cointegrated/rubert-tiny2-cedr-emotion-detection")

text = "Добавьте муку и перемешайте - вы сами удивитесь, насколько все быстро произойдет!"
results = classifier(text)
print(results)
```

Вышеуказанный код выполнит классификацию текста и выведет результат на экран.

## Пример
`ввод: "Добавьте муку и перемешайте - вы сами удивитесь , насколько все быстро произойдет !"`

`вывод: 'label': 'surprise', 'score': 0.9651550054550171`


# Дашков Артем Андреевич src/topic_is_bad_good_or_neutral.py

Это приложение предназначено для определения эмоциональной окраски текста. Используется модель blanchefort/rubert-base-cased-sentiment, обученная на 351.797 текстах, что дает неплохой результат. Приложение легко в использовании - после его запуска в окне "Текст:" достаточно ввести тестируемую фразу и нажать Enter! Приложение определит эмоциональную окраску текста, а также вероятность правильности ответа. Всего доступно три уровня эмоциальной окраски текста, так текст может быть:
1. позитивным;
2. нейтральным;
3. негативным.

Для выхода из приложения в окне "Текст:" необходимо ввести кодовое слово "Выход".

Ниже представлены примеры использования программы:

```
Привет! Ниже вам необходимо ввести текст для определения его эмоциональной окраски. Для выхода из приложениядостаточно в поле 'Текст' ввести слово 'выход'.
Текст: У лукоморья дуб зелёный; Златая цепь на дубе том: И днём и ночью кот учёный Всё ходит по цепи кругом;
Фраза негативная с вероятностью 75.15%

Текст: Идёт направо — песнь заводит, Налево — сказку говорит. Там чудеса: там леший бродит, Русалка на ветвях сидит;
Фраза нейтральная с вероятностью 67.99%

Текст: Там на неведомых дорожках Следы невиданных зверей; Избушка там на курьих ножках Стоит без окон, без дверей;
Фраза негативная с вероятностью 75.16%

Текст: выход
Выполнен выход из приложения.

```