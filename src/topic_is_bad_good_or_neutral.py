from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      "blanchefort/rubert-base-cased-sentiment")

print("Привет! Ниже вам необходимо ввести текст для определения его эмоциональной окраски. Для выхода из приложения"
      "достаточно в поле 'Текст' ввести слово 'выход'.")
text = input ("Текст: ").lower()

while text != "выход":
    result = classifier(text)[0]
    if result.get('label') == 'POSITIVE':
        print(f"Фраза позитивная с вероятностью {result.get('score'):.2%} \n")
    elif result.get('label') == 'NEUTRAL':
        print(f"Фраза нейтральная с вероятностью {result.get('score'):.2%} \n")
    else:
        print(f"Фраза негативная с вероятностью {result.get('score'):.2%} \n")

    text = input ("Текст: ").lower()
print("Выполнен выход из приложения.")