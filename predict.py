import transformers
from transformers import pipeline
from data_loader import title_dataset

"""TEST ON EXAMPLE"""

text = title_dataset['test'][5]['text']
text = "summarize: " + text
print(text)

summarizer = pipeline("summarization", model="t5_small_title_generator")
pred = summarizer(text, max_length=15, min_length=5)

print(pred)