from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from test1_eval import load_json, write_json as save_json, raw_datasets
from eval import format_text, run_eval


def do_post_process(query):
    query = query.replace("?"," ?")
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


def calculate_bleu(n_):
    prefix = "./json/"
    filename = "test_3_T0"+str(n_)+"_diversity_gpt.json"
    data = load_json(prefix + filename)

    cleaned_sparql = [do_post_process(s) for s in data["generated_sparql"]]
    sparql = data["sparql"]
    # questions = data1["questions"]
    bleu = []
    for i, query in enumerate(cleaned_sparql):
        gold = format_text(sparql[i])
        generated = format_text(query)
        bleu_c = sentence_bleu([gold.split()], generated.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        bleu_4 = sentence_bleu([gold.split()], generated.split(), weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
        print(round(bleu_c, 4), round(bleu_4, 4))
        bleu.append([round(bleu_c, 4), round(bleu_4, 4)])

    data["template"] = "T0"+str(n_)
    data["cleaned_sparql"] = cleaned_sparql

    data["bleu"] = bleu

    save_json("test_3_T0"+str(n_)+"_diversity_gpt_results_cleaned.json", data)


def get_best():
    data = {}

    for i in range(8):
        filename = "test_3_T0"+str(i+1)+"_diversity_gpt_results_cleaned.json"
        data["T0"+str(i+1)] = load_json(filename)
    keys = [x for x in data]
    len_ = len(data.get("T01").get("cleaned_sparql"))
    # save_json("test_3_diversity_dolly_results_cleaned.json", data)
    print (keys)
    questions = []
    generated_sparql = []
    cleaned_sparql = []
    bleu_scores = []
    winner_template = []
    same_template = []
    n_same_template = 0
    sparql = data.get("T01").get("sparql")
    question_template = data.get("T01").get("question_template")
    for i in range(len_):
        q_cand = [data.get(x).get("bleu_c")[i] for x in keys]
        print(q_cand)
        q_cand_score = max(q_cand)
        bleu_scores.append(q_cand_score)
        key = keys[q_cand.index(q_cand_score)]
        print(key)
        winner_template.append(key)
        same_template.append(key==question_template[i])
        if key==question_template[i]:
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
    print (n_same_template)
    print(sum(same_template))
    save_json("test_3_diversity_gpt_results_cleaned_final.json", test_3_data)


def correggi():
    data = {}
    for i in range(8):
        filename = "test_3_T0" + str(i + 1) + "_diversity_gpt_results_cleaned.json"
        res = load_json(filename)
        data["template"] = res["template"]
        data["questions"] = res["questions"]
        data["sparql"] = res["sparql"]
        data["generated_sparql"] = res["generated_sparql"]
        data["cleaned_sparql"] = res["cleaned_sparql"]
        data["question_template"] = res["suggestions"]
        data["bleu_c"] = [x[0] for x in res["bleu"]]
        data["bleu_4"] = [x[1] for x in res["bleu"]]
        save_json(filename, data)


if __name__ == '__main__':
    pass
    # for i in range(8):
    #     calculate_bleu(i+1)
    # correggi()
    get_best()