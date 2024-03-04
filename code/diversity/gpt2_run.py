import json
import torch
import random
from transformers import pipeline, AutoTokenizer

gpt2 = pipeline('text-generation', model="./gpt2_ep10/", max_new_tokens=384, device=0 if torch.cuda.is_available() else "cpu", return_full_text=False)
tokenizer = AutoTokenizer.from_pretrained("./gpt2_ep10/")


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


def load_json(file__name):
    try:
        data_file = open(file__name, "r", encoding='utf-8')
        file_data = json.loads(data_file.read())
        data_file.close()
        return file_data
    except FileNotFoundError:
        return None


def clean(st):
    st = st.replace("\n", " ")
    st = st.replace("?", " ?")
    st = st.replace("{", " { ")
    st = st.replace("}", " } ")
    st = st.replace("\\'", "'")

    while "  " in st:
        st = st.replace("  ", " ")
    return st


def main(filename):
    data = load_json(filename)

    q_list = data["questions"]
    suggestions = data["suggestions"]
    gs = data["generated_sparql"]
    sparql = data["sparql"]
    lens = data.get("prompt_len")
    if lens is None:
        lens = []

    print(len(q_list))

    for i, question in enumerate(q_list[len(gs):]):
        print(i, end=" ")
        res_ = tokenizer.encode(question)
        len_ = len(res_)
        lens.append(len_)
        print("len: ", len_)
        if len_ > 600:
            question = tokenizer.decode(res_[-600:])
            len_ = 600

        if len_ <= 600:
            res = gpt2(question)
            gs.append(res[0]["generated_text"])
            result = {"questions": q_list, "sparql": sparql, "generated_sparql": gs, "prompt_len": lens,
                      "suggestions": suggestions}
            save_json(filename[:-5] + "_complete.json", result)


main("test_2_ce_ft_gpt2.json")