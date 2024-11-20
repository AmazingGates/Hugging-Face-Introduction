from chat import MiniCPMVChat, img2base64
import torch
import json

torch.manual_seed(0)

chat_model = MiniCPMVChat('openbmb/MiniCPM-Llama3-V-2_5')

im_64 = img2base64('./assets/airplane.jpeg')

# First round chat 
msgs = [{"role": "user", "content": "Tell me the model of this aircraft."}]

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Introduce something about Airbus A380."})

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)