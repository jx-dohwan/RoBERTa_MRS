from transformers import RobertaForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn

class FineModel(nn.Module):
    def __init__(self):
        super(FineModel, self).__init__()
        self.model = RobertaForMaskedLM.from_pretrained('klue/roberta-base')
        self.hiddenDim = self.model.config.hidden_size
        
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        special_tokens = {'sep_token': '<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))        
        
        """ score matrix """
        # self.W = nn.Linear(self.hiddenDim, 3)
        self.W2 = nn.Linear(self.hiddenDim, 2)
        
    def forward(self, batch_input_tokens, batch_input_attentions):
        ## Binary Classification (pointwise)
        outputs = self.model(batch_input_tokens, attention_mask=batch_input_attentions, output_hidden_states=True)['hidden_states'][-1] # [B, L, hidden_dim]
        cls_outputs = outputs[:,0,:] # [B, hidden_dim]
        cls_logits = self.W2(cls_outputs) # [B, 2]
         
        return cls_logits