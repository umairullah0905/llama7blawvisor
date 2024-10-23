from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

file = open("inference.txt", "r")
content = file.read()

def preprocess_case(text):
    max_tokens = 1000
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

input = content
input = preprocess_case(input)


model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token= "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")
tokenizer = AutoTokenizer.from_pretrained(model_id)


def create_prompt(text):
  
    prompt = f"""
    ### Instructions:
    Predict the outcome of the case proceeding as 'rejected' or 'accepted' \
    and subsequently provide an explanation based on significant sentences in the case proceedings.

    Format:
    Case decision:
    Explanation:

    ### Input:
    case_proceeding: <{text}>

    ### Response:"""

    return prompt


case_pro=input
prompt = create_prompt(case_pro)
input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=512,)
output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
print(output)