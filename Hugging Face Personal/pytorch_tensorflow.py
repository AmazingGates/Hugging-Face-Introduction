from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# In this section we will see how we can combine our code with Pytorch or Tensorflow

# We will start by applying the pipeline like we did before.

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis",model = model, tokenizer = tokenizer)

# Now we will be using multiple sentences as opposed to the single sentence that we were using previously.

# We will be using a list for our sentences.

X_train = ["I love my Wife more than she'll ever know.",
           "I think the Barbies feeling her too."]

# We store this in our X_train.

# Then we feed the X_train to our classifier pipeline, which is stored in a variable (res).

# Then we print the variable (res)

res = classifier(X_train)
print(res)



# Now separately, we will call the tokenizer with the X_train data.

# We will call this our batch, and we will also pass in some parameters.

# return_tensors="pt" indicates that the return will be in pytorch format.

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

# Next we will do the inference in pytorch.

# So for this we say with torch no grad.

# Then we call our model and that is where we unpack our batch, because it is a dictionary.

# And then we can apply different functions, like f.softmax, to get the predictions, or torch.argmax, to get the 
#labels

# And again, these predictions should be the same scores that we get from the classifier(pipeline) because it 
#essentially is the same step except that now we do it for ourselves.

# Now we can run this to see our results.

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)

    # 