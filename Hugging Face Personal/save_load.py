from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# In this section we will go over the steps of savng and loading tokenizers.

# To save a tokenizer and model we can specify a save directory.

# Then we can call tokenizer.save pretrained and also model.save pretrained

save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# When we want to load them again we can pick a class like we did below and call AutoTokenizer from pretrained

tok = AutoTokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassification.from_pretrained(save_directory)