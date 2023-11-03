# SE_G1_2
Группа 1.2 по предмету Программная инженерия

#Драчев Алексей Анатольевич aleksrf@gmail.com
Добавлен файл src/scale_of_emotions.py
---------------------------------------
Данный файл предназначен для ввода текста на английском языке и 
последующий перевод данного текста на французкий, при условии, что оценка по "шкале эмоций" выше чем 50%
При недостаточной оценки перевод не осуществляется

Для работы данного файла требуется установить библиотеку Hugging Face
!pip install transformers sentencepiece

Замечание: Ntrcn ljk;ty ,snm yf ghfdbkmyjv fyukbqcrjv zpsrt? lfyysq fkujhbnv yt ghjdthztn ghfdbkmyjcnm fyukbqcrjuj? bcgjkmpeqnt lkz  'njuj lheubt ,b,kbjntrb/'

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


