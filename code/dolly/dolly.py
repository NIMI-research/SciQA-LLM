import json
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch
from transformers import pipeline

dolly = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
question = '''
{'English question': 'Indicate the model that performed best in terms of Accuracy metric on the Natural Questions benchmark dataset?', 
 'sparql query': 'SELECT DISTINCT ?model ?model_lbl WHERE { ?metric a orkgc:Metric; rdfs:label ?metric_lbl. FILTER (str( ?metric_lbl) = "Accuracy") { SELECT ?model ?model_lbl WHERE { ?dataset a orkgc:Dataset; rdfs:label ?dataset_lbl. FILTER (str( ?dataset_lbl) = "Natural Questions") ?benchmark orkgp:HAS_DATASET ?dataset; orkgp:HAS_EVALUATION ?eval. ?eval orkgp:HAS_VALUE ?value; orkgp:HAS_METRIC ?metric. ?cont orkgp:HAS_BENCHMARK ?benchmark; orkgp:HAS_MODEL ?model. ?model rdfs:label ?model_lbl. } ORDER BY DESC( ?value) LIMIT 1 } } '} 
 use the example to return a sparql query for the following question: Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?
'''

res = dolly(question)
print(res[0]["generated_text"])



quit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            queries.append("'" + question + "' translated to sparql would be: ")
        else:
            final_q = "'" + suggestion[1] + "' translated to sparql is '" + suggestion[0] + "' \n '" + question + "' translated to sparql would be: "
            queries.append(final_q)
    return queries

query_list = prepare_queries()
print(len(query_list))

# print (len(not_generated))
# print(not_generated)
# print(json.dumps(patterns))

n = 10
q_list = list(divide_chunks(query_list, n))
sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

gs = []
gst = []
i = 0

for group in q_list:
    print(str(i)+"%", end="  ")
    i += 2
    print(type(group))
    res = dolly(group)
    print(res)
    # print(res[0]["generated_text"])

result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "generated_with_special_tokens": gst}

with open("dolly_results.json", "w", encoding="utf-8") as text_file:
    print(json.dumps(result), file=text_file)