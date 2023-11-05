from transformers import pipeline

classifier = pipeline("text-classification", "cointegrated/rubert-tiny2-cedr-emotion-detection")

results = classifier("Добавьте муку и перемешайте - вы сами удивитесь , насколько все быстро произойдет !")

print(results)
