import torch
import torch.nn as nn 
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pdb

import os
def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'post_model.bin'))

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

from dataset import post_loader
from torch.utils.data import DataLoader

data_path = './korean_smile_style_dataset/smile.csv'
post_dataset = post_loader(data_path)
post_dataloader = DataLoader(post_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=post_dataset.collate_fn)

from model import PostModel
post_model = PostModel().cuda()

training_epochs = 1 # colab에서 돌아가게끔 하기 위해 1을 사용, 실습자는 5로 해서 사용하면 됨
max_grad_norm = 10
lr = 1e-5
num_training_steps = len(post_dataset)*training_epochs
num_warmup_steps = len(post_dataset)
optimizer = torch.optim.AdamW(post_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

for epoch in range(training_epochs):
    print(f"{epoch}번째 학습시작!!")
    post_model.train() 
    for i_batch, data in enumerate(tqdm(post_dataloader)):
        batch_corrupt_tokens, batch_output_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_urc_labels, batch_mlm_attentions, batch_urc_attentions = data
        batch_corrupt_tokens = batch_corrupt_tokens.cuda()
        batch_output_tokens = batch_output_tokens.cuda()
        batch_urc_inputs = batch_urc_inputs.cuda()
        batch_urc_labels = batch_urc_labels.cuda()
        batch_mlm_attentions = batch_mlm_attentions.cuda()
        batch_urc_attentions = batch_urc_attentions.cuda()
        
        """Prediction"""
        corrupt_mask_outputs, urc_cls_outputs = post_model(batch_corrupt_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_mlm_attentions, batch_urc_attentions)
        #pdb.set_trace()        

        """Loss calculation & training"""
        original_token_indexs = []
        for i_batch in range(len(batch_corrupt_mask_positions)):
            original_token_index = []
            batch_corrupt_mask_position = batch_corrupt_mask_positions[i_batch]
            for pos in batch_corrupt_mask_position:
                original_token_index.append(batch_output_tokens[i_batch,pos].item())
            original_token_indexs.append(original_token_index)
    
        mlm_loss = 0
        for corrupt_mask_output, original_token_index in zip(corrupt_mask_outputs, original_token_indexs):
            mlm_loss += CELoss(corrupt_mask_output, torch.tensor(original_token_index).cuda())        
        urc_loss = CELoss(urc_cls_outputs, batch_urc_labels)
        
        loss_val = mlm_loss + urc_loss
        
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(post_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()   
        
SaveModel(post_model, '.')