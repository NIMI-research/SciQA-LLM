import json
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
prefix = "translate English to Sparql: "


checkpoint = "T5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
new_words = ['{', '}']
tokenizer.add_tokens(new_words)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,return_dict=True).to(device)
model.resize_token_embeddings(len(tokenizer))

raw_datasets = load_dataset("orkg/SciQA")
print(raw_datasets)

def divide_chunks(l_, n_):
    for i_ in range(0, len(l_), n_):
        yield l_[i_:i_ + n_]

def clean(st):
    st = st.replace("\n", " ")
    st = st.replace("?", " ?")
    st = st.replace("{", " { ")
    st = st.replace("}", " } ")
    st = st.replace("\\'", "'")

    while "  " in st:
        st = st.replace("  ", " ")
    return st


def get_key(q):
    t0 = q.get('template_id')
    if t0 is None:
        t0 = "None"
    t = str(q.get("number_of_patterns")) + "-" + t0
    return t


def get_keys():
    train = raw_datasets.get("train")
    patterns = {}
    for q in train:
        t =  get_key(q)
        query = clean(q["query"]["sparql"])
        question = q["question"]["string"]
        if t not in patterns or len(query) > len(patterns.get(t)):
            patterns[t] = [query, question]
    return patterns

def prepare_queries():
    keys = get_keys()
    data = raw_datasets.get("test")
    queries = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = keys.get(t)
        if suggestion is None:
            print("Error with key", t)
            queries.append(prefix + question)
        else:
            final_q = prefix + suggestion[1] + " = " + suggestion[0] + "\n" + prefix + question
            queries.append(final_q)
    return queries

query_list = prepare_queries()
print(len(query_list))

# print (len(not_generated))
# print(not_generated)
# print(json.dumps(patterns))

n = 10

q = list(divide_chunks(query_list, n))
sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

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

result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "generated_with_special_tokens": gst}

with open("t5_base_results.json", "w", encoding="utf-8") as text_file:
    print(json.dumps(result), file=text_file)