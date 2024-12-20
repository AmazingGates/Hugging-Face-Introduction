# In this section we will go over how we can use different models from the model hub.

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


QA_input = {
    'question': 'Why is Goku stronger than Superman?',
    'context': 'Although many believe that Superman is stronger, the feats and achievements reached by Goku are on another level all together, Goku is the ultimate warrior!'
}

QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(res)