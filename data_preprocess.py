from data_loader import title_dataset
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(item):

  labels = tokenizer(text=item["title"], max_length=30, truncation=True)
  inputs = tokenizer(text=item["text"], max_length=530, truncation=True)

  model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


tokenized_title_dataset = title_dataset.map(preprocess_function, batched=True)