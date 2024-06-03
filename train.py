#library for loading dataset (not absolute path) 
import os
import numpy as np
import transformers
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# libs for evaluation
import evaluate
import rouge_score
rouge = evaluate.load("rouge")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

from data_preprocess import tokenized_title_dataset

"""
FOR EVALUATION
"""

def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
   prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
   result["gen_len"] = np.mean(prediction_lens)
   return {k: round(v, 4) for k, v in result.items()}


"""
PARAMETERS FOR MODEL
"""


model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

output_dir = "t5_small_title_generator"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
   output_dir= output_dir,
   evaluation_strategy="epoch",
   learning_rate=2e-5,
   per_device_train_batch_size=2,
   per_device_eval_batch_size=2,
   weight_decay=0.01,
   save_total_limit=3,
   num_train_epochs=3,
   predict_with_generate=True,
   fp16=False,  # Disable mixed precision training
)

"""
MODEL TRAIN
"""

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_title_dataset["train"],
   eval_dataset=tokenized_title_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

"""
MODEL SAVE
"""

trainer.save_model(output_dir)

"""
MODEL EVALUATE on eval_dataset parameter
"""

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

"""
TEST
"""

print("TEST RESULTS")

test_results = trainer.evaluate(eval_dataset=tokenized_title_dataset['test'])
print(test_results)
