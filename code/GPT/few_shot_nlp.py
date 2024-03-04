import random

import json
import time

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datasets import load_dataset
from gpt import gpt_query
threshold = 0.25

model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else "cpu")
raw_datasets = load_dataset("orkg/SciQA")
print(raw_datasets)
embed_data = torch.load('train_embeddings_m.pt')


def load_json(file__name):
    try:
        data_file = open(file__name, "r", encoding='utf-8')
        file_data = json.loads(data_file.read())
        data_file.close()
        return file_data
    except FileNotFoundError:
        return None


def save_json(filename,data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


def get_similar_test_a(element, items=None, embeddings=None, num=None, reversed_=False):
    emb_items = None

    if items is None and embeddings is not None:
        emb_items = embeddings["emb_questions"]
        items = embeddings["keys"]
    elif items is not None:
        emb_items = model.encode(items)

    if len(element) == 0 or emb_items is None:
        return []

    emb_element = model.encode(element)

    result = []
    scores = cos_sim(emb_element, emb_items)

    if num is None or num < 2:
        maximus = torch.max(scores, 1)
        m = float(maximus.values[0])
        i = int(maximus.indices[0])
        if m > threshold:
            result = [[round(m, 4), items[i], embeddings["questions"][i], embeddings["queries"][i]]]
        return result
    else:
        scored_texts = []
        for i, score in enumerate(scores[0]):
            scored_texts.append(
                [round(score.item(), 4), items[i], embeddings["questions"][i], embeddings["queries"][i]])
        sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)

        keys = []
        samples = []
        for sample in sorted_scored_texts:
            if sample[1] not in keys:
                samples.append(sample)
                keys.append(sample[1])

        samples = samples[:num]
        if reversed_:
            samples.reverse()
        return samples


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

    result = []
    scores = cos_sim(emb_element, emb_items)

    if num is None or num < 2:
        maximus = torch.max(scores, 1)
        m = float(maximus.values[0])
        i = int(maximus.indices[0])
        if m > threshold:
            result = [[round(m, 4), items[i], embeddings["questions"][i], embeddings["queries"][i]]]
        return result
    else:
        scored_texts = []
        for i, score in enumerate(scores[0]):
            scored_texts.append([round(score.item(), 4), items[i], embeddings["questions"][i], embeddings["queries"][i]])
        sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)
        return sorted_scored_texts[:num]


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


def get_random(n_):
    train = raw_datasets.get("train")
    sample = random.sample(list(train), n_)
    sample_list = []
    for q in sample:
        t = get_key(q)
        query = clean(q["query"]["sparql"])
        question = q["question"]["string"]
        sample_list.append([query, question, t])
    return sample_list


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


def save_embedding_m():
    train = raw_datasets.get("train")

    train_m = [q for q in train if not q.get("auto_generated")]
    print(train_m)

    questions = [q["question"]["string"] for q in train_m]
    queries = [clean(q["query"]["sparql"]) for q in train_m]
    keys = [get_key(q) for q in train_m]
    embeddings = {}
    emb_questions = model.encode(questions)
    embeddings["questions"] = questions
    embeddings["emb_questions"] = emb_questions
    embeddings["queries"] = queries
    embeddings["keys"] = keys
    torch.save(embeddings, 'train_embeddings_m.pt')
    return embeddings



def get_similar_test_b(element, items=None, embeddings=None, num=None, reversed_=False):
    emb_items = None

    if items is None and embeddings is not None:
        emb_items = embeddings["emb_questions"]
        items = embeddings["keys"]
    elif items is not None:
        emb_items = model.encode(items)

    if len(element) == 0 or emb_items is None:
        return []

    emb_element = model.encode(element)

    result = []
    scores = cos_sim(emb_element, emb_items)

    if num is None or num < 2:
        maximus = torch.max(scores, 1)
        m = float(maximus.values[0])
        i = int(maximus.indices[0])
        if m > threshold:
            result = [[round(m, 4), items[i], embeddings["questions"][i], embeddings["queries"][i]]]
        return result
    else:
        scored_texts = []
        for i, score in enumerate(scores[0]):
            scored_texts.append(
                [round(score.item(), 4), items[i], embeddings["questions"][i], embeddings["queries"][i]])
        sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)

        key = sorted_scored_texts[0][1]
        samples = []
        for sample in sorted_scored_texts:
            if sample[1] == key:
                samples.append(sample)

        samples = samples[:num]
        if reversed_:
            samples.reverse()
        return samples


def prepare_queries(n_, reversed_=False, method=None):
    data = raw_datasets.get("test")
    data = [q for q in data if not q.get("auto_generated")]
    queries = []
    suggestions = []

    if n_ == 0:
        for q in data:
            question = q["question"]["string"]
            final_q = "what is the sparql query for:  " + question
            queries.append(final_q)
    else:
        for q in data:
            t = get_key(q)
            question = q["question"]["string"]

            if method == "test_a":
                suggestion = get_similar_test_a(question, embeddings=embed_data, num=n_, reversed_=reversed_)
            elif method == "test_b":
                suggestion = get_similar_test_b(question, embeddings=embed_data, num=n_, reversed_=reversed_)
            else:
                suggestion = get_similar(question, embeddings=embed_data, num=n_)
            suggestions.append([[[x[0], x[1]] for x in suggestion],t])

            if suggestion is None or len(suggestion)==0:
                print("Error with key", t)
                queries.append("translate the following English text '" + question + "' to a sparql query")
            else:
                final_q = ""
                for i_, k in enumerate(suggestion):
                    final_q += "\n input (English text): " + k[2]
                    final_q += "\n output (Sparql query): " + k[3]

                # works better with gpt
                final_q += "\n with this example what is the sparql query for:  " + question

                # works better with dolly
                # final_q += "\n input (English text): " + question
                # final_q += "\n output (Sparql query): "
                queries.append(final_q)

    return queries, suggestions


def prepare_random_queries(n_):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = get_random(n_)
        suggestions.append([[x[2] for x in suggestion], t])
        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for i_, k in enumerate(suggestion):
                final_q += "\n input (English text): " + k[1]
                final_q += "\n output (Sparql query): " + k[0]
            final_q += "\n with this example what is the sparql query for:  " + question
            queries.append(final_q)
    return queries, suggestions


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



def prepare_queries_cheating(n_):
    keys = get_keys(n_)
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = keys.get(t)
        suggestions.append(t)
        if suggestion is None:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for i_, k in enumerate(suggestion):
                final_q += "\n input (English text): " + k[1]
                final_q += "\n output (Sparql query): " + k[0]

            # works better with dolly
            # final_q += "\n input (English text): " + question
            # final_q += "\n output (Sparql query): "

            # works better with gpt
            final_q += "\n with this example what is the sparql query for:  " + question
            queries.append(final_q)

    return queries, suggestions


def main(shots=1, prefix="nlp_GPT_", method="nlp"):
    try:
        filename = prefix+str(shots)+"_shot_results.json"
        result = load_json(filename)
        if result is None:
            if method == "nlp":
                query_list, suggestions = prepare_queries(shots)
            elif  method == "test_a" or method == "test_b":
                query_list, suggestions = prepare_queries(shots, method=method)
            elif  method == "random":
                query_list, suggestions = prepare_random_queries(shots)
            else:
                query_list, suggestions = prepare_queries_cheating(shots)

            print(len(query_list))
            sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]
            gs = []

        else:
            query_list=result.get("questions")
            suggestions = result.get("suggestions")
            sparql = result.get("sparql")
            gs = result.get("generated_sparql")

        for query in query_list[len(gs):]:
            # if len(gs)>=20:
            #     break
            gq = gpt_query(query)
            time.sleep(5)
            gq = clean(gq)
            gs.append(gq)
            print(gq)
            result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "suggestions": suggestions}
            save_json(filename, result)
            # break
    except Exception as e:
        print("Error:", e)
        main(shots=shots, method=method, prefix=prefix)


def runner(filename):
    try:
        result = load_json(filename)
        if result is None:
            return
        else:
            query_list = result.get("questions")
            suggestions = result.get("suggestions")
            sparql = result.get("sparql")
            gs = result.get("generated_sparql")

        for query in query_list[len(gs):]:
            gq = gpt_query(query)
            time.sleep(2)
            gq = clean(gq)
            gs.append(gq)
            print(gq)
            result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "suggestions": suggestions}
            save_json(filename, result)
            # break
    except Exception as e:
        print("Error:", e)
        runner(filename)


if __name__ == '__main__':
    runner("gpt_ddp_data.json")
# save_embedding_m()

# main(shots=3, method="test_a", prefix="test_a_")
# main(shots=5, method="test_a", prefix="test_a_")
# main(shots=7, method="test_a", prefix="test_a_")

# main(shots=3, method="test_b", prefix="test_b_")
# main(shots=5, method="test_b", prefix="test_b_")
# main(shots=7, method="test_b", prefix="test_b_")
# main(shots=1, prefix="nlp_GPT4_")
# main(shots=3, prefix="nlp_GPT4_")
# main(shots=5, prefix="nlp_GPT4_")
# main(shots=7, prefix="nlp_GPT4_")

# print(prepare_queries(7))
# main(shots=1, method="random", prefix="random_")
# main(shots=3, method="random", prefix="random_")
# main(shots=5, method="random", prefix="random_")
# main(shots=7, method="random", prefix="random_")
# main(shots=7, prefix="nlp_GPT-3.5_hand_")