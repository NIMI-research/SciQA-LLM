import json

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prefix = "translate English to Sparql: "
tokenizer = AutoTokenizer.from_pretrained("en2sparql_T5_model")
model = AutoModelForSeq2SeqLM.from_pretrained("en2sparql_T5_model").to(device)

# books = load_dataset("json", data_files={'test':'test.json'})
books = load_dataset("orkg/SciQA")
print(books["test"])

queries = []
sparql = []

for feature in books["test"]:
    query = prefix + feature.get("question").get("string")
    queries.append(query)
    gold_sparql = feature.get("query").get("sparql")
    sparql.append(gold_sparql)

print(len(queries))

def divide_chunks(l_, n_):
    # looping till length l
    for i_ in range(0, len(l_), n_):
        yield l_[i_:i_ + n_]

n = 10

q = list(divide_chunks(queries, n))

gs = []
gst = []
i = 0

for group in q:
    print(str(i)+"%", end="  ")
    i += 2
    inputs = tokenizer(group, max_length=512, truncation=True, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=30, top_p=0.95)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_texts2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    generated_texts2 = [x.replace("<pad>", "").replace("</s>", "").strip() for x in generated_texts2]

    gs += generated_texts
    gst += generated_texts2

result = {"questions": queries, "sparql": sparql, "generated_sparql": gs, "generated_with_special_tokens": gst}

with open("my_results.json", "w", encoding="utf-8") as text_file:
    print(json.dumps(result), file=text_file)

