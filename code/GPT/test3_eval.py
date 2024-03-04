import csv
import json

from datasets import load_dataset
# raw_datasets = load_dataset("orkg/SciQA")


def write_csv(filename_,content):
    with open(filename_, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(content)


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file_data = json.loads(data_file.read())
    data_file.close()
    return file_data


def write_json(file__name, content):
    with open(file__name, "w", encoding="utf-8") as text_file:
        print(json.dumps(content), file=text_file)



def main(filename="test_3_mult_2_diversity_gpt_results_cleaned_final.json", file_to_save="test_3_mult_diversity_gpt_final.json", csv_file="test_3_mult_2_diversity_gpt_results_cleaned.csv"):


    data = load_json(filename)
    cleaned_sparql = data["cleaned_sparql"]

    sparql = []
    predicted = []
    questions = []
    for i, item in enumerate(cleaned_sparql):
        if item.startswith("SELECT"):
            predicted.append(item)
            sparql.append(data["sparql"][i])
            questions.append(data["questions"][i])

    data["cleaned_sparql"] = cleaned_sparql
    # data["questions"] = data1["questions"]
    data["c_questions"] = questions
    data["c_sparql"] = sparql
    data["c_predicted"] = predicted

    write_json(file_to_save, data)

    print(len(predicted))

    from eval import run_eval
    unfair = run_eval(predicted,sparql)
    fair = run_eval(data["cleaned_sparql"],data["sparql"])

    rows = [["Questions","SPARQL in Gold Standard", "SPARQL generated",
             "exact match"]]

    equals = 0

    for i, question in enumerate(data["questions"]):
        row = [question, data["sparql"][i], data["cleaned_sparql"][i],
               data["cleaned_sparql"][i].replace(" ", "") == data["sparql"][i].replace(" ", "")
               ]
        if row[3] is True:
            equals +=1
        # print(row)
        rows.append(row)

    rows.append([])
    rows.append(["Metrics on query only"])
    for key in unfair:
        rows.append([key,unfair.get(key)])

    rows.append([])
    rows.append(["Metrics on all rows"])
    for key in fair:
        rows.append([key,fair.get(key)])
    rows.append([])
    rows.append(["Number of sparql queries generated equal to given", equals])
    rows.append([])
    rows.append(["Generated text starts with 'SELECT'", len(predicted)])

    write_csv(csv_file,rows)
    print("n. of equals: ", equals)

if __name__ == '__main__':
    main("test_3_mult_1_diversity_dolly_cleaned.json","test_3_mult_1_diversity_dolly_final.json","test_3_mult_1_diversity_dolly.csv")