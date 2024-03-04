import torch
import json
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")

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
        code = sorted(code, key=lambda x: x[2], reverse=False)
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
            final_q = ""
            for i_, k in enumerate(suggestion):
                final_q += "\n input (English text): " + k[1]
                final_q += "\n output (Sparql query): " + k[0]
            # works better with gpt
            # final_q += "\n with this example what is the sparql query for:  " + question

            # works better with dolly
            final_q += "\n input (English text): " + question
            final_q += "\n output (Sparql query): "

            queries.append(final_q)
    return queries


def save_json(filename,data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


shots = 8
query_list = prepare_queries(shots)

print(len(query_list))

n = 20
q_list = list(divide_chunks(query_list, n))
sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

gs = []
lens =[]

i = 0

for group in q_list:
    print(str(i) + "%", end="  ")
    i += 1/len(q_list)*100

    res_ = [tokenizer.encode(question) for question in group]
    len_ = [len(x) for x in res_]
    warning = [x for x in len_ if x > 2048]
    if len(warning) > 0:
        print(warning)
        quit()
    lens += len_

    res = dolly(group)
    print(res)
    gst = [x[0]["generated_text"] for x in res]

    for ii, l in enumerate(gst):
        for iii in range(5):
            if "SELECT" not in l:
                print(iii,ii)
                res = dolly(group[ii])
                gst[ii] = res[0]["generated_text"]
                l = gst[ii]
            else:
                break
    gs += gst

    result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "prompt_len": lens}
    save_json("dolly_"+str(shots)+"_shot_results_tok.json", result)
    break




