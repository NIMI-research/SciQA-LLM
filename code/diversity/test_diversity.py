import json
import torch
from eval import run_eval

embed_data = torch.load('train_embeddings.pt')

def load_json(file__name, path="./json/"):
    data_file = open(path + file__name, "r", encoding='utf-8')
    file_data = json.loads(data_file.read())
    data_file.close()
    return file_data


def save_json(filename,data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


if __name__ == '__main__':
    a = load_json("nlp_dolly_7_shot_results_cleaned.json")
    # print(a["questions"][1])
    # print([key for key in embed_data])
    # print(embed_data["questions"][0])
    # print(embed_data["emb_questions"][0])
    # print(embed_data["keys"][0])
    keys = []
    for key in embed_data["keys"]:
        if key not in keys:
            keys.append(key)
    print(keys)

    rows = []
    for i, key in enumerate(embed_data["keys"]):
        if "None" in key:
            rows.append((i,key))
    # print(rows)

    gold = []
    generated = []
    for i, s in enumerate(a["suggestions"]):
        if "None" not in s[1]:
            generated.append(a["cleaned_sparql"][i])
            gold.append(a["sparql"][i])
    run_eval(generated, gold)
