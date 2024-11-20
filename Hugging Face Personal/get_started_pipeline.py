from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've finally decided to take charge and learn Hugging Face outside of the course")

print(res)