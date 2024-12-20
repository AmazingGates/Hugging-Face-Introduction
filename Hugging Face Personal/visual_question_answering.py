# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
#model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
#model.eval()

image = Image.open('Delia 1.png').convert('RGB')

plt.imshow(image)
plt.axis("off")
plt.show

question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
#res = model.chat(
#    image=image,
#    msgs=msgs,
#    tokenizer=tokenizer,
#    sampling=True,
#    temperature=0.7,
#    stream=True
#)

#generated_text = ""
#for new_text in res:
#    generated_text += new_text
#    print(new_text, flush=True, end='')
