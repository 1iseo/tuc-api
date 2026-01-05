import torch
from typing import Optional, Tuple
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os

class JointBERT(nn.Module):
    def __init__(self, model_name, num_intents, num_slots, dropout_rate=0.1):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        
        self.bert = AutoModel.from_config(self.config)
        
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, num_intents)
        )
        self.slot_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, num_slots)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        return intent_logits, slot_logits

print("JointBERT model class defined!")

# 2. SETUP APP & LOAD MODEL
app = FastAPI(title="Campus QA Bot API")

model = None
tokenizer = None
intent2idx = {}
idx2intent = {}
slot2idx = {}
idx2slot = {}
config = {}

def load_artifacts():
    global model, tokenizer, intent2idx, idx2intent, slot2idx, idx2slot, config
    
    deploy_dir = "deployment"
    
    # Load configuration
    with open(os.path.join(deploy_dir, "label_config.json"), "r") as f:
        config = json.load(f)
        
    intent2idx = config["intent2idx"]
    slot2idx = config["slot2idx"]
    idx2intent = {int(k): v for k, v in enumerate(config["intent_labels"])}
    idx2slot = {int(k): v for k, v in enumerate(config["slot_labels"])}

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(deploy_dir, "tokenizer"))

    # Initialize Model Structure
    print("Initializing model...")
    model_cpu = JointBERT(
        model_name=config["model_name"],
        num_intents=len(config["intent_labels"]),
        num_slots=len(config["slot_labels"])
    )

    # Apply Quantization Structure (Crucial step for loading quantized weights)
    # We must match the quantization command used in training exactly
    print("Applying dynamic quantization structure...")
    model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Load Weights
    print("Loading weights...")
    state_dict = torch.load(os.path.join(deploy_dir, "model_quantized.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

load_artifacts()

class QueryRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: QueryRequest):
    text = request.text
    max_seq_len = config["max_seq_len"]

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids))

    # Inference
    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids)

    # Post-processing
    intent_probs = torch.softmax(intent_logits, dim=-1)
    intent_idx = torch.argmax(intent_probs, dim=-1).item()
    intent_confidence = intent_probs[0, intent_idx].item()
    
    slot_preds = torch.argmax(slot_logits, dim=-1)[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    slot_labels = [idx2slot[idx] for idx in slot_preds]

    # Extract Entities logic (Simplified for API)
    entities = []
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, slot_labels):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if label.startswith('B-'):
            if current_entity:
                entities.append({"entity": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith('I-') and current_entity == label[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append({"entity": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append({"entity": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})

    return {
        "text": text,
        "intent": idx2intent[intent_idx],
        "confidence": round(intent_confidence, 4),
        "entities": entities
    }

@app.get("/")
def health_check():
    return {"status": "ok", "model": "JointBERT Campus Bot"}
