import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import List, Optional, Union
import numpy as np

class PolymerTransformer(nn.Module):
    def __init__(self, num_tasks: int = 5, pretrained_model: str = "seyonec/PubChem10M_SMILES_BPE_450k"):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Load pre-trained ChemBERTa model
        self.transformer = RobertaModel.from_pretrained(pretrained_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        
        # Freeze some layers for transfer learning
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        
        # Add task-specific layers
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_tasks)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific prediction
        logits = self.classifier(pooled_output)
        return logits

class PolymerTransformerModel:
    def __init__(self, num_tasks: int = 5, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = PolymerTransformer(num_tasks=num_tasks).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        
    def tokenize_smiles(self, smiles_list: List[str]):
        """Tokenize SMILES strings for the transformer."""
        return self.model.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def train_step(self, batch_smiles: List[str], targets: torch.Tensor):
        """Train the model for one step."""
        self.model.train()
        
        # Tokenize SMILES
        encoded = self.tokenize_smiles(batch_smiles)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """Make predictions for a list of SMILES strings."""
        self.model.eval()
        predictions = []
        
        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            encoded = self.tokenize_smiles(batch_smiles)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                predictions.append(outputs.cpu().numpy())
        
        return np.vstack(predictions)
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 