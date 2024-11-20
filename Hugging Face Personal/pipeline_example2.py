from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "My goal is to be the best Computer Programmer",
    max_length=30,
    num_return_sequences=2
)

print(res)