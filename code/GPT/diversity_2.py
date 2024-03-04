import json
import time

from gpt import gpt_query


def load_json(file__name):
    try:
        data_file = open(file__name, "r", encoding='utf-8')
        file_data = json.loads(data_file.read())
        data_file.close()
        return file_data
    except FileNotFoundError:
        return None


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as json_file:
        print(json.dumps(data), file=json_file)


def clean(st):
    st = st.replace("\n", " ")
    st = st.replace("?", " ?")
    st = st.replace("{", " { ")
    st = st.replace("}", " } ")
    st = st.replace("\\'", "'")

    while "  " in st:
        st = st.replace("  ", " ")
    return st


def main():
    try:
        filename = "test_3_mult_diversity_gpt.json"
        data = load_json(filename)
        if data is None:
            print("File non trovato!")
            quit()

        else:
            for key in data:
                result = data[key]
                query_list = result.get("questions")
                suggestions = result.get("suggestions")
                sparql = result.get("sparql")
                gs = result.get("generated_sparql")

                for query in query_list[len(gs):]:
                    gq = gpt_query(query)
                    time.sleep(5)
                    gq = clean(gq)
                    gs.append(gq)
                    print(gq)
                    result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs,
                              "suggestions": suggestions}
                    data[key] = result
                    save_json(filename, data)
                    # if len(gs)>19:
                    #     break
    except Exception as e:
        print("Error:", e)
        main()


if __name__ == '__main__':
    main()
