from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from test1_eval import load_json, write_json as save_json, raw_datasets
from eval import format_text, run_eval


def do_post_process(query):
    query = query.replace("?", " ?")
    query = query.replace("\n", " ")

    while "  " in query:
        query = query.replace("  ", " ")

    while "--" in query:
        query = query.replace("--", "-")

    query = query.replace("|\n+-+\n|", " ")

    if "SELECT" in query:
        index = query.index("SELECT")
        query = query[index:]
    query = query.replace("- what is the sparql query for '", "")
    for ss in raw_datasets["test"]:
        question_ = ss["question"]["string"]
        query = query.replace(question_, "")
    return query


def calculate_bleu(data, key):
    cleaned_sparql = [do_post_process(s) for s in data["generated_sparql"]]
    sparql = data["sparql"]
    bleu_c_list = []
    bleu_4_list = []

    for i, query in enumerate(cleaned_sparql):
        gold = format_text(sparql[i])
        generated = format_text(query)
        bleu_c = sentence_bleu([gold.split()], generated.split(), weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=SmoothingFunction().method1)
        bleu_4 = sentence_bleu([gold.split()], generated.split(), weights=(0, 0, 0, 1),
                               smoothing_function=SmoothingFunction().method1)
        bleu_c_list.append(round(bleu_c, 4))
        bleu_4_list.append(round(bleu_4, 4))

    data["template"] = key
    data["cleaned_sparql"] = cleaned_sparql
    data["bleu_c"] = bleu_c_list
    data["bleu_4"] = bleu_4_list
    return data


def get_best(filename="test_3_mult_diversity_gpt.json",
             file_to_save="test_3_mult_2_diversity_gpt_results_cleaned_final.json"):
    data = load_json(filename)
    keys = [x for x in data if not x.endswith("2") and not x.endswith("1")]
    # keys = [x for x in data if not x.endswith("2")]
    # keys = [x for x in data]
    len_ = len(data.get(keys[0]).get("cleaned_sparql"))

    print(keys)
    questions = []
    generated_sparql = []
    cleaned_sparql = []
    bleu_scores = []
    winner_template = []
    same_template = []
    n_same_template = 0
    sparql = data.get(keys[0]).get("sparql")
    question_template = data.get(keys[0]).get("templates")
    for i in range(len_):
        q_cand = [data.get(x).get("bleu_c")[i] for x in keys]
        # print(q_cand)
        q_cand_score = max(q_cand)
        bleu_scores.append(q_cand_score)
        key = keys[q_cand.index(q_cand_score)]
        print(key.split("_")[0])
        winner_template.append(key)
        same_template.append(key.split("_")[0] == question_template[i])
        print(question_template[i])
        if key.split("_")[0] == question_template[i]:
            n_same_template += 1
        questions.append(data.get(key).get("questions")[i])
        generated_sparql.append(data.get(key).get("generated_sparql")[i])
        cleaned_sparql.append(data.get(key).get("cleaned_sparql")[i])
    test_3_data = {
        "questions": questions,
        "sparql": sparql,
        "question_template": question_template,
        "generated_sparql": generated_sparql,
        "cleaned_sparql": cleaned_sparql,
        "bleu_scores": bleu_scores,
        "winner_template": winner_template,
        "same_template": same_template
    }
    print(n_same_template)
    print(sum(same_template))
    save_json(file_to_save, test_3_data)


def main(filename="test_3_mult_diversity_gpt.json"):
    complete_data = load_json(filename)
    for key in complete_data:
        complete_data[key] = calculate_bleu(complete_data[key], key)
    save_json(filename, complete_data)


if __name__ == '__main__':
    pass
    # main("test_3_mult_diversity_dolly.json")
    get_best("test_3_mult_diversity_dolly.json", "test_3_mult_1_diversity_dolly_cleaned.json")
