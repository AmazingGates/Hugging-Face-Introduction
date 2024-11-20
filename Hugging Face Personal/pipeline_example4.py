# Here we will take a deeper look at pipelines by lookng at Tokenizers and model classes.

# AutoTokenize is a generic class, and AutoModelForSequenceClassification is also a generic class but is more specified
#for the sequence classification.

# But there are specific Tokenizers we can use for specific task.

# We can get a better understanding of these topics by taking a look at the documentation.

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I love my Wife more than she'll ever know.")

#print(res)

# Next we can create instances of the new classes that we imported.

model_name = "distilbert-base-uncased-finetuned-sst-2-english" # This is how specified the model we want
model = AutoModelForSequenceClassification.from_pretrained(model_name) # This is how call the model class and 
#specify that we want it from pretrained
tokenizer = AutoTokenizer.from_pretrained(model_name) # We will do the same thing for the AutoTokenizer. Also,
#this from_pretrained method is very important in huggingface that we will see a lot of times, so we should get
#familiar with it.

# Now that we have this, we can copy and paste the code from above, and now for the pipeline we can say 
#model = model, and tokenizer = tokenizer

# And since the model we are using is the same as the default model, we should have the same result as we had 
#when we ran the original model.

classifier = pipeline("sentiment-analysis",model = model, tokenizer = tokenizer)

res = classifier("I love my Wife more than she'll ever know.")

#print(res)


# More useful tokenizer information.

# A tokenizer puts a text in a matehematical representation that the model understands.

# And in order to use this we can call the tokenizer directly and give it a text as input or we can also 
#put in multiple texts as ones as a list.

# So now we can do this and print it.

tokenizer = AutoTokenizer.from_pretrained(model_name)

sequence = "Using a Transformer network is simple"

res = tokenizer(sequence)

print(res)

# And we can also do this separately so we can call tokenizer.tokenizer.

tokens = tokenizer.tokenize(sequence)
print(tokens)

# This will give us tokens back, then we can call tokenizer.convert tokens to ids.

# This willl give us the ids.

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# And we can do it the other way around by calling tokenizer.decode(ids)

# This will give us the original string back when we print decoded_string.

decoded_string = tokenizer.decode(ids)
print(decoded_string)

# So if we apply the tokenizer directly we get a dictionary. And the dict contains the input ids that look like this.
#{'input_ids': [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102].

# Then we also have an attention mask, so for now we don't have to worry about this because the attention mask is
#bascically just a list of zeros and ones. And a zero means that the attention layer should ignore this token.
#'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

# Then if we do this separately we call tokenizer.tokenize, then we see the different tokens 
#['using', 'a', 'transform', '##er', 'network', 'is', 'simple']

# Then if we convert the tokens to ids then each token has a unique corresponding id like we see here.
#[2478, 1037, 10938, 2121, 2897, 2003, 3722]

# And if we decode this we get back the original string back, but please note that we basically removed the
#capitalization.
#using a transformer network is simple

# Also note that if we compare the input ids to the token ids, (see lines 78 and 88), we notice we have the 
#same exact numbers. The only difference comes on line 78 where we have two additional numbers, 101 and 102. These
#two numbers represent the beginning of the sentence and the end of the sentence.

# This is how a tokenizer works.


# 