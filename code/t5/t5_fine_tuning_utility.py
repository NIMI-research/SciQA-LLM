from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np

checkpoint = "T5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
new_words = ['{', '}']
tokenizer.add_tokens(new_words)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,return_dict=True)
model.resize_token_embeddings(len(tokenizer))

prefix = "translate English to Sparql: "
metric = evaluate.load("sacrebleu")


def prepare_dataset():
    raw_datasets = load_dataset("orkg/SciQA")
    print(raw_datasets)
    train = raw_datasets["train"]
    validation = raw_datasets["validation"]
    test = raw_datasets["test"]
    assert train.features.type == validation.features.type == test.features.type
    query_dataset = concatenate_datasets([train, validation, test])
    query_dataset = query_dataset.train_test_split(test_size=500)
    query_dataset["train"].to_json("train.json")
    query_dataset["test"].to_json("test.json")


def preprocess_function(examples):
    inputs = [prefix + string["string"] for string in examples["question"]]
    targets = [ string["sparql"] for string in examples["query"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
