from transformers import AutoTokenizer, AutoModel
import torch

def get_smiles_embedding(smiles, model_name='seyonec/ChemBERTa-zinc-base-v1'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(smiles, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding 