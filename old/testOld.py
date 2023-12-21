# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

# pip install -q accelerate peft bitsandbytes transformers trl tensorboardx -U

import torch
from transformers import pipeline

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
model_name = "/home/ankur/Documents/trainTransformer/ankur-llama2-1k/"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "What is NSC?"
outputs = pipe(f"<s>[INST] {prompt} [/INST]")
print(outputs[0]["generated_text"])
