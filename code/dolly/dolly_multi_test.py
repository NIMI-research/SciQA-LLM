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


def get_keys(n_):
    train = raw_datasets.get("train")
    patterns = {}
    for q in train:
        t = get_key(q)
        query = clean(q["query"]["sparql"])
        question = q["question"]["string"]
        if t not in patterns:
            patterns[t] = [[query, question, len(query)]]
        else:
            patterns[t].append([query, question, len(query)])

    for t in patterns:
        code = patterns.get(t)
        code = sorted(code, key=lambda x: x[2], reverse=True)
        patterns[t] = code[:n_]
    return patterns


def prepare_queries(n_):
    keys = get_keys(n_)
    data = raw_datasets.get("test")
    queries = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = keys.get(t)
        if suggestion is None:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = "" #"The following are example of generating sparql query from English text."
            for i_, k in enumerate(suggestion):
                final_q += "\n input (English text): " + k[1]
                final_q += "\n output (Sparql query): " + k[0]
                # final_q += " - '" + k[1] + "' translated to a sparql query is " + k[0]
            # final_q += " - translate the following English question '" + question + "' to a sparql query "
            final_q += "\n input (English text): " + question
            final_q += "\n output (Sparql query): "

            queries.append(final_q)
    return queries


shots = 6
query_list = prepare_queries(shots)

print(len(query_list))


n = 5
q_list = list(divide_chunks(query_list, n))
sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

gs = []

i = 0

for group in q_list:
    print(str(i) + "%", end="  ")
    i += 1/len(q_list)*100
    print(type(group))
    res = dolly(group)
    print(res)
    gst = [x[0]["generated_text"] for x in res]
    # print(res[0]["generated_text"])
    gs += gst
    # break

result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs}

with open("dolly_"+str(shots)+"_shot_results.json", "w", encoding="utf-8") as text_file:
    print(json.dumps(result), file=text_file)