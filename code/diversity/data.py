import random

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from transformers import pipeline
import json

from datasets import load_dataset

threshold = 0.25

# dolly = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else "cpu")
embed_data = torch.load('train_embeddings.pt')
raw_datasets = load_dataset("orkg/SciQA")
print(raw_datasets)


def save_embedding():
    train = raw_datasets.get("train")
    questions = [q["question"]["string"] for q in train]
    queries = [clean(q["query"]["sparql"]) for q in train]
    keys = [get_key(q) for q in train]
    embeddings = {}
    emb_questions = model.encode(questions)
    embeddings["questions"] = questions
    embeddings["emb_questions"] = emb_questions
    embeddings["queries"] = queries
    embeddings["keys"] = keys
    torch.save(embeddings, 'train_embeddings.pt')
    return embeddings


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file_data = json.loads(data_file.read())
    data_file.close()
    return file_data


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


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
    # t = str(q.get("number_of_patterns")) + "-" + t0
    return t0


def get_key_c(q):
    t0 = q.get('template_id')
    if t0 is None:
        t0 = "None"
    t = str(q.get("number_of_patterns")) + "-" + t0
    return t0


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
    ordered_keys = [x for x in patterns]
    ordered_keys.sort()
    for t in ordered_keys:
        code = patterns.get(t)
        code = sorted(code, key=lambda x: x[2], reverse=False)
        patterns[t] = code[:n_]
        print(code[:n_])
    return patterns

def get_similar(element, items=None, embeddings=None, num=None):
    emb_items = None

    if items is None and embeddings is not None:
        emb_items = embeddings["emb_questions"]
        items = embeddings["keys"]
    elif items is not None:
        emb_items = model.encode(items)

    if len(element) == 0 or emb_items is None:
        return []

    emb_element = model.encode(element)
    scores = cos_sim(emb_element, emb_items)

    scored_texts = []
    for i, score in enumerate(scores[0]):
        scored_texts.append(
            [round(score.item(), 4), items[i], embeddings["questions"][i], embeddings["queries"][i]])

    scored_texts = [x for x in scored_texts if x[1] != "None"]
    sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)

    return sorted_scored_texts[:num]


def prepare_queries(elements, _model):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = get_similar(question, embeddings=embed_data, num=1)

        suggestions.append([[[x[0], x[1]] for x in suggestion], t])
        template = suggestion[0][1]
        suggestion = elements.get(template)

        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for k in elements.get(template):
                if "ft_gpt2" in _model:
                    # works better with fine-tuned gpt2?
                    final_q += "<|endoftext|>" + k[1] + " "
                    final_q += k[0]
                else:
                    final_q += "\n input (English text): " + k[1]
                    final_q += "\n output (Sparql query): " + k[0]

            if "gpt3" in _model:
                # works better with gpt
                final_q += "\n with this example what is the sparql query for:  " + question

            elif "ft_gpt2" in _model:
                # works better with fine-tuned gpt2?
                final_q += "<|endoftext|>" + question
            else:
                # works better with dolly and other pt models
                final_q += "\n input (English text): " + question
                final_q += "\n output (Sparql query): "

            queries.append(final_q)

    return queries, suggestions


def prepare_queries_2(elements, _model):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        template = random.choice(list(elements))
        suggestions.append([template, t])
        suggestion = elements.get(template)
        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for k in suggestion:
                if "ft_gpt2" in _model:
                    # works better with fine-tuned gpt2?
                    final_q += "<|endoftext|>" + k[1] + " "
                    final_q += k[0]
                else:
                    final_q += "\n input (English text): " + k[1]
                    final_q += "\n output (Sparql query): " + k[0]

            if "gpt3" in _model:
                # works better with gpt
                final_q += "\n with this example what is the sparql query for:  " + question

            elif "ft_gpt2" in _model:
                # works better with fine-tuned gpt2?
                final_q += "<|endoftext|>" + question
            else:
                # works better with dolly and other pt models
                final_q += "\n input (English text): " + question
                final_q += "\n output (Sparql query): "

            queries.append(final_q)

    return queries, suggestions



def prepare_queries_3(element,_model):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestions.append(t)
        suggestion = element
        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for k in suggestion:
                if "ft_gpt2" in _model:
                    # works better with fine-tuned gpt2?
                    final_q += "<|endoftext|>" + k[1] + " "
                    final_q += k[0]
                else:
                    final_q += "\n input (English text): " + k[1]
                    final_q += "\n output (Sparql query): " + k[0]

            if "gpt3" in _model:
                # works better with gpt
                final_q += "\n with this example what is the sparql query for:  " + question

            elif "ft_gpt2" in _model:
                # works better with fine-tuned gpt2?
                final_q += "<|endoftext|>" + question
            else:
                # works better with dolly and other pt models
                final_q += "\n input (English text): " + question
                final_q += "\n output (Sparql query): "

            queries.append(final_q)

    return queries, suggestions



def prepare_data_test_1(_model):
    keys = get_keys(1)
    del keys["None"]
    query_list, suggestions = prepare_queries(keys, _model)
    sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]
    result = {"questions": query_list, "sparql": sparql, "generated_sparql": [],
              "suggestions": suggestions}
    save_json(_model+"_test_1_ce.json", result)



def prepare_data_test_2(_model):
    keys = get_keys(1)
    del keys["None"]
    query_list, suggestions = prepare_queries_2(keys, _model)
    sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]
    result = {"questions": query_list, "sparql": sparql, "generated_sparql": [],
              "suggestions": suggestions}
    save_json(_model+"_test_2_ce.json", result)


def prepare_data_test_3(_model,n="1"):
    keys = get_keys(1)
    del keys["None"]

    for key in keys:
        query_list, suggestions = prepare_queries_3(keys[key], _model)
        sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]
        result = {"questions": query_list, "sparql": sparql, "generated_sparql": [],
                  "suggestions": suggestions, "template": key}
        save_json("test_3_" + key + "_" + n + "_diversity_" + _model+".json", result)



def prepare_queries_3_mult(element,_model, n_):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for i, q in enumerate(data):
        if i>= n_:
            break
        t = get_key(q)
        question = q["question"]["string"]
        suggestions.append(t)
        suggestion = element
        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for k in suggestion:
                if "ft_gpt2" in _model:
                    # works better with fine-tuned gpt2?
                    final_q += "<|endoftext|>" + k[1] + " "
                    final_q += k[0]
                else:
                    final_q += "\n input (English text): " + k[1]
                    final_q += "\n output (Sparql query): " + k[0]

            if "gpt3" in _model:
                # works better with gpt
                final_q += "\n with this example what is the sparql query for:  " + question

            elif "ft_gpt2" in _model:
                # works better with fine-tuned gpt2?
                final_q += "<|endoftext|>" + question
            else:
                # works better with dolly and other pt models
                final_q += "\n input (English text): " + question
                final_q += "\n output (Sparql query): "

            queries.append(final_q)

    return queries, suggestions




def prepare_data_test_3_mult(_model,n):
    keys = get_keys(3)
    del keys["None"]
    data = {}
    for key in keys:
        print(key, keys[key])
        for i, element in enumerate(keys[key]):
            query_list, suggestions = prepare_queries_3_mult([element], _model, n)
            sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")][:n]
            result = {"questions": query_list, "sparql": sparql, "generated_sparql": [],
                      "templates": suggestions, "template": key+"_"+str(i)}
            data[key+"_"+str(i)] = result
    save_json(_model +"_test_3_mult_diversity.json", data)


if __name__ == '__main__':
    # save_embedding()
    prepare_data_test_1("pt_gpt2")
    prepare_data_test_2("pt_gpt2")
    # prepare_data_test_3("pt-gpt2")
    # prepare_data_test_3("dolly")
    prepare_data_test_3_mult("pt_gpt2", 513)

