from transformers import pipeline

prefix = "translate English to Sparql: "
# text = "Can you provide links to code used in papers that benchmark the Orthogonalized Soft VSM model?"
# text = "Can you list the models that have been evaluated on the SentEval dataset?"
text="What is the best performing model benchmarking the STL-10 dataset in terms of Percentage correct metric?"
# translator = pipeline("translation", model="en2sparql_T5_model")
# print(translator(prefix+text))


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("en2sparql_T5_model")
inputs = tokenizer(prefix+text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("en2sparql_T5_model")
model.resize_token_embeddings(len(tokenizer))
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=30, top_p=0.95)
# print(outputs)
# out=tokenizer.decode(outputs[0], skip_special_tokens=False)
# print(out)

text = "SELECT DISTINCT ?metric ?metric_lbl WHERE { ?dataset a orkgc:Dataset; rdfs:label ?dataset_lbl. FILTER (str(?dataset_lbl) = 'BUCC Russian-to-English') ?benchmark orkgp:HAS_DATASET ?dataset; orkgp:HAS_EVALUATION ?eval. OPTIONAL {?eval orkgp:HAS_METRIC ?metric. ?metric rdfs:label ?metric_lbl.} }"
tokenized = tokenizer(text, max_length=512, truncation=True)

detokenized = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=False)
print(detokenized)