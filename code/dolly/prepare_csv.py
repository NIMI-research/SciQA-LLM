import csv
import json


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file = json.loads(data_file.read())
    data_file.close()
    return file

def write_csv(filename,content):
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(content)

results = load_json("dolly_0_shot_results.json")
rows = [["Questions","SPARQL in Gold Standard", "SPARQL generated",
         # "Cleaned",
         "exact match"]]
for i, question in enumerate(results["questions"]):
    row = [question, results["sparql"][i], results["generated_sparql"][i],
           # results["cleaned_sparql"][i],
           results["generated_sparql"][i] == results["sparql"][i]]
    # question = question.replace("translate English to Sparql: ", "").replace("\\'", "'")
    # row.append(results["generated_with_special_tokens"][i])
    print(row)
    rows.append(row)

write_csv("dolly_0_shot.csv",rows)
from eval import run_eval
run_eval(results["generated_sparql"],results["sparql"])