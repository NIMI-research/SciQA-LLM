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


def save_json(filename,data):
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


def main(prefix="./json/test_", test="2"):
    try:
        filename = prefix + test + "_diversity_gpt.json"
        result = load_json(filename)
        if result is None:
            print("File non trovato!")
            quit()

        else:
            query_list=result.get("questions")
            suggestions = result.get("suggestions")
            sparql = result.get("sparql")
            gs = result.get("generated_sparql")

        for query in query_list[len(gs):]:
            gq = gpt_query(query)
            time.sleep(5)
            gq = clean(gq)
            gs.append(gq)
            print(gq)
            result = {"questions": query_list, "sparql": sparql, "generated_sparql": gs, "suggestions": suggestions}
            save_json(filename, result)
            if len(gs)>19:
                break
    except Exception as e:
        print("Error:", e)
        main(test=test, prefix=prefix)

if __name__ == '__main__':
    main(test="3_T01")
    main(test="3_T02")
    main(test="3_T03")
    main(test="3_T04")
    main(test="3_T05")
    main(test="3_T06")
    main(test="3_T07")
    main(test="3_T08")