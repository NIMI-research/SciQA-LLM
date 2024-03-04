import openai


OPENAI_API_KEY = "the key"
openai.organization = "the organization"

openai.api_key = OPENAI_API_KEY


def gpt_query(msg, context=None, adding="\n you must return only the Sparql query!"):
    if context is None:
        context = "You are a translator from natural language to Sparql using the provided examples."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": context},
            {"role": "user", "content": msg + adding}],
        temperature=0.5,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        timeout=30
    )
    return response['choices'][0]['message']['content']

