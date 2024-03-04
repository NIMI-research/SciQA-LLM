import json
import jsonlines

import tiktoken
from datasets import load_dataset


def clean(st):
    st = st.replace("\n", " ")
    st = st.replace("?", " ?")
    st = st.replace("{", " { ")
    st = st.replace("}", " } ")
    st = st.replace("\\'", "'")

    while "  " in st:
        st = st.replace("  ", " ")
    return st


def save_jsonl_file():
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    raw_datasets = load_dataset("orkg/SciQA")
    print(raw_datasets)

    examples = []
    length = 0
    for example in raw_datasets.get("train"):
        context = 'You are a translator from natural language to Sparql'
        question = example["question"]["string"]
        sparql = clean(example["query"]["sparql"])
        # length += len(encoding.encode(context + " " + question + " " + sparql))

        sample = {'messages': [{'role': 'system',
                                'content': context},
                               {'role': 'user',
                                'content': question},
                               {'role': 'assistant',
                                'content': sparql}]}
        examples.append(sample)
        sample_len = len(encoding.encode(json.dumps(sample)))
        length += sample_len

    print("Total tokens:", length)
    print ("Total cost $", round(length*0.008*5/1000, 2))
    with jsonlines.open('train.jsonl', mode='w') as writer:
        for sample in examples:
            writer.write(sample)



import openai
import os
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# openai.File.create(
#   file=open("test.jsonl", "rb"),
#   purpose='fine-tune'
# )

# with jsonlines.open("test.jsonl", mode='r') as reader:
#     for obj in reader:
#         print(obj)

# with jsonlines.open('output.jsonl', mode='w') as writer:
#     writer.write(...)

if __name__ == '__main__':
    save_jsonl_file()