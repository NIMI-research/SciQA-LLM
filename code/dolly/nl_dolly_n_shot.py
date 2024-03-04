import json
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

threshold = 0.25

model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else "cpu")
# model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else "cpu")
raw_datasets = load_dataset("orkg/SciQA")
print(raw_datasets)
embed_data = torch.load('train_embeddings.pt')
dolly = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")


def divide_chunks(l_, n_):
    for i_ in range(0, len(l_), n_):
        yield l_[i_:i_ + n_]


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


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
            scored_texts.append(
                [round(score.item(), 4), items[i], embeddings["questions"][i], embeddings["queries"][i]])
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


def prepare_queries(n_):
    data = raw_datasets.get("test")
    queries = []
    suggestions = []
    for q in data:
        t = get_key(q)
        question = q["question"]["string"]
        suggestion = get_similar(question, embeddings=embed_data, num=n_)
        suggestions.append([[[x[0], x[1]] for x in suggestion], t])

        if suggestion is None or len(suggestion) == 0:
            print("Error with key", t)
            queries.append("translate the following English text '" + question + "' to a sparql query")
        else:
            final_q = ""
            for i_, k in enumerate(suggestion):
                final_q += "\n input (English text): " + k[2]
                final_q += "\n output (Sparql query): " + k[3]

            # works better with gpt
            # final_q += "\n with this example what is the sparql query for:  " + question

            # works better with dolly
            final_q += "\n input (English text): " + question
            final_q += "\n output (Sparql query): "
            queries.append(final_q)
    return queries, suggestions


def main(shots=3, attempts=10, batch=50):
    query_list, suggestions = prepare_queries(shots)
    print(len(query_list))

    n = batch
    q_list = list(divide_chunks(query_list, n))
    sparql = [clean(x["query"]["sparql"]) for x in raw_datasets.get("test")]

    gs = []
    lens = []
    i = 0

    for group in q_list:
        print(str(i) + "%", end="  ")
        i += 1 / len(q_list) * 100

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
            for iii in range(attempts):
                if "SELECT" not in l:
                    print(iii, ii)
                    res = dolly(group[ii])
                    gst[ii] = res[0]["generated_text"]
                    l = gst[ii]
                else:
                    break
        gs += gst

        result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "prompt_len": lens,
                  "suggestions": suggestions}
        save_json("nlp_dolly_" + str(shots) + "_shot_results_tok.json", result)


if __name__ == '__main__':
    # save_embedding()
    main()