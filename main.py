from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Model
model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token= "...........")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Preprocessing
def preprocess_case(text):
    max_tokens = 4000
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

# prompt
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

def create_prompt1(text):
  
    prompt = f"""
    ### Instructions:
    You are an expert in Indian law, and your task is to suggest the appropriate section of the Indian Constitution or Indian laws, including the Indian Penal Code (IPC), under which someone can file a case based on the case described. For each scenario, provide a brief explanation of the legal section, the potential consequences for the offender, and any additional related sections that may apply.

    Format:
    Explanation:

    ### Input:
    case: <{text}>
    

    ### Response:"""

    return prompt

class StringInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict_case(data: StringInput):
    if len(data.text) > 50000000:  # Example limit, adjust as necessary
        raise HTTPException(status_code=400, detail="Input text is too long. Please limit to 5000 characters.")
    try:
        # Preprocess input text
        input_text = preprocess_case(data.text)
        prompt = create_prompt(input_text)

        # Prepare input for the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.to(device)

        # Generate prediction
        outputs = model.generate(input_ids=input_ids, max_new_tokens=512)

        # Decode the output
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_output = result[len(prompt):].strip()  # Remove the prompt from the output

        return {"prediction": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new/")
async def predict_case(data: StringInput):
    if len(data.text) > 50000000:  # Example limit, adjust as necessary
        raise HTTPException(status_code=400, detail="Input text is too long. Please limit to 5000 characters.")
    try:
        # Preprocess input text
        input_text = preprocess_case(data.text)
        prompt = create_prompt1(input_text)

        # Prepare input for the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.to(device)

        # Generate prediction
        outputs = model.generate(input_ids=input_ids, max_new_tokens=512)

        # Decode the output
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_output = result[len(prompt):].strip()  # Remove the prompt from the output

        return {"prediction": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root route for testing
@app.get("/")
async def root():
    return {"message": "The FastAPI server is running!"}
