import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class DistilBERTPolicyEncoder(nn.Module):
    def __init__(self, proj_dim=1024, pretrained_model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.output_proj = nn.Linear(self.bert.config.hidden_size, proj_dim)
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, input_texts):
        """
        Args:
            input_texts: List[str] or a batch of tokenized inputs (dict)

        Returns:
            Tensor of shape (B, 1, proj_dim)
        """
        if isinstance(input_texts, list):
            inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        else:
            inputs = input_texts  # already tokenized

        inputs = {k: v.to(self.output_proj.weight.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (B, L, H)

        cls_token = last_hidden_state[:, 0]  # take [CLS] token as summary
        projected = self.output_proj(cls_token)  # (B, proj_dim)

        return projected.unsqueeze(1)  # match shape (B, 1, D)
