from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from t5_fine_tuning_utility import preprocess_function, tokenizer, checkpoint, compute_metrics, model


books = load_dataset("orkg/SciQA")
print(books)
tokenized_books = books.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="en2sparql_T5_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=30,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()


