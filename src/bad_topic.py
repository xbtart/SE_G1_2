from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
import torch 
import numpy as np

tokenizer = BertTokenizer.from_pretrained('apanc/russian-sensitive-topics')
model = BertForSequenceClassification.from_pretrained('apanc/russian-sensitive-topics')

import json
with open("id2topic.json") as f:
    target_vaiables_id2topic_dict = json.load( f)

# Запрос ввода текста с клавиатуры
input_text = input("Введите текст на русском: ")

tokenized = tokenizer.batch_encode_plus([input_text], max_length = 512, truncation=True, return_token_type_ids=False)
tokens_ids,mask = torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask']) 
with torch.no_grad():
    model_output = model(tokens_ids,mask)

def adjust_multilabel(y, is_pred = False):
    y_adjusted = []
    for y_c in y:
        y_test_curr = [0]*19
        index = str(int(np.argmax(y_c)))
        y_c = target_vaiables_id2topic_dict[index]
    return y_c

preds = adjust_multilabel(model_output['logits'], is_pred = True)

# Вывод результата
if (preds == 'none'):
    print('Плохая тема не затронута.')
else:
    print(f"Распознаны следующие острые темы, {preds}!")