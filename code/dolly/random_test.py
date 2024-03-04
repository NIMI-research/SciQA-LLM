import csv
import json
import os.path
from eval import run_eval


def load_json(file__name):
    data_file = open(file__name, "r", encoding='utf-8')
    file_data = json.loads(data_file.read())
    data_file.close()
    return file_data


def write_json(file__name, content):
    with open(file__name, "w", encoding="utf-8") as text_file:
        print(json.dumps(content), file=text_file)


def run_tests(shots=1, threshold=0.1):
    results = {}
    prefix = "json/"
    filename = prefix + "random_dolly_" + str(shots) + "_shot_results_cleaned.json"
    if not os.path.isfile(filename):
        return
    data = load_json(filename)
    # print(len(data["suggestions"]))
    results["shots"] = shots
    results["Generated"] = len(data["suggestions"])
    results["threshold"] = threshold
    sim_mean = 0
    generated = []
    sparql = []
    for i, sample in enumerate(data["suggestions"]):
        # print(sample)
        elem_sim = 0
        r_list = sample[0]
        r = sample[1]
        for r_elem in r_list:
            if r_elem == r:
                sim_mean += 1 / len(r_list)
                elem_sim += 1 / len(r_list)
                # print(r_elem, r)
        if elem_sim >= threshold:
            generated.append(data["cleaned_sparql"][i])
            sparql.append(data["sparql"][i])
    results["samples_above_threshold"] = len(generated)
    results["rdm_mean_n"] = sim_mean
    results["rdm_mean_%"] = sim_mean / len(data["suggestions"]) * 100
    print("len: ", len(generated))
    metrics = run_eval(generated, sparql)
    for key in metrics:
        results[key] = metrics.get(key)
    return results


def write_csv(filename_, content):
    with open(filename_, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(content)


def main():
    res = []
    for i in range(8):
        if i == 0:
            continue
        res.append(run_tests(shots=i))
    print(json.dumps(res))
    table = {}
    for row in res:
        if row is None:
            continue
        for key in row:
            if key in table:
                table[key].append(row.get(key))
            else:
                table[key] = [row.get(key)]
    print(table)
    rows = []
    for key in table:
        rows.append([key] + table.get(key))
    write_csv("random_test_results.csv", rows)


main()
