import csv
import json


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file = json.loads(data_file.read())
    data_file.close()
    return file

results = load_json("results.json")
rows = [["Questions","SPARQL in Gold Standard", "SPARQL generated","exact match"]]
for i, question in enumerate(results["questions"]):
    row = []
    question = question.replace("translate English to Sparql: ", "").replace("\\'", "'")
    row.append(question)
    row.append(results["sparql"][i])
    row.append(results["generated_sparql"][i])
    # row.append(results["generated_with_special_tokens"][i])
    row.append(results["generated_sparql"][i] == results["sparql"][i])
    print(row)
    rows.append(row)

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(rows)