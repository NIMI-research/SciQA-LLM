import json


def write_json(file__name, content):
    with open(file__name, "w", encoding="utf-8") as text_file:
        print(json.dumps(content), file=text_file)


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file_data = json.loads(data_file.read())
    data_file.close()
    return file_data

filename1 = "reversed_test_A_nlp_dolly_7_shot_results_tok.json"
filename2 = "old_reversed_test_A_nlp_dolly_7_shot_results_tok.json"

complete_data = load_json(filename2)
data = load_json(filename1)

complete_data["generated_sparql"] = complete_data["generated_sparql"][:200] + data["generated_sparql"]
complete_data["prompt_len"] = complete_data["prompt_len"][:200]+data["prompt_len"]
lens = [len(complete_data[x]) for x in complete_data]
print(lens)
write_json("complete_reversed_test_A_nlp_dolly_7_shot_results_tok.json", complete_data)
