import torch
import json
from transformers import pipeline
from datasets import load_dataset

dolly = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
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
        t = get_key(q)
        query = clean(q["query"]["sparql"])
        question = q["question"]["string"]
        if t not in patterns or len(query) > len(patterns.get(t)):
            patterns[t] = [query, question]
    return patterns


def prepare_queries():

    data = raw_datasets.get("test")
    queries = []
    for q in data:
        question = q["question"]["string"]
        queries.append(" translate to a sparql query the following English question:" + question)
    return queries



query_list = prepare_queries()
print(len(query_list))

n = 5
q_list = list(divide_chunks(query_list, n))
sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

gs = []

i = 0

for group in q_list:
    print(str(i) + "%", end="  ")
    i += len(group)/len(q_list)*100
    res = dolly(group)
    print(res)
    gst = [x[0]["generated_text"] for x in res]
    gs += gst

result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs}

with open("dolly_0_shot_results.json", "w", encoding="utf-8") as text_file:
    print(json.dumps(result), file=text_file)