""" 데이터 로딩 """
from dataset import fine_loader
from torch.utils.data import DataLoader
train_path ='/content/drive/MyDrive/인공지능/멀티턴응답선택/korean_smile_style_dataset/train.json'
train_dataset = fine_loader(train_path)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)

dev_path = '/content/drive/MyDrive/인공지능/멀티턴응답선택/korean_smile_style_dataset/dev.json'
dev_dataset = fine_loader(dev_path)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dev_dataset.collate_fn)
""" 모델 로딩 """
import torch
from model import FineModel
fine_model = FineModel().cuda()
fine_model.load_state_dict(torch.load('/content/drive/MyDrive/인공지능/멀티턴응답선택/post_model.bin'), strict=False)

""" loss 식 """
import torch.nn as nn
def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

""" 평가 """
from torch.nn.functional import softmax
def CalP1(fine_model, dataloader):
    fine_model.eval()
    correct = 0
    for i_batch, data in enumerate(tqdm(dataloader, desc="evaluation")):
        batch_input_tokens, batch_input_attentions, batch_input_labels = data
        
        batch_input_tokens = batch_input_tokens.cuda()
        batch_input_attentions = batch_input_attentions.cuda()
        batch_input_labels = batch_input_labels.cuda()
        
        """Prediction"""
        outputs = fine_model(batch_input_tokens, batch_input_attentions)    
        probs = softmax(outputs, 1)
        true_probs = probs[:,1]
        pred_ind = true_probs.argmax(0).item()
        gt_ind = batch_input_labels.argmax(0).item()
        
        if pred_ind == gt_ind:
            correct += 1
    return round(correct/len(dataloader)*100, 2)

""" 모델 저장 """
import os
def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, '/content/drive/MyDrive/인공지능/멀티턴응답선택/fine_model.bin'))

""" 학습 """
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pdb

training_epochs = 5
max_grad_norm = 10
lr = 1e-6
num_training_steps = len(train_dataset)*training_epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(fine_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

best_p1 = 0
for epoch in range(training_epochs):
    fine_model.train() 
    for i_batch, data in enumerate(tqdm(train_dataloader)):
        batch_input_tokens, batch_input_attentions, batch_input_labels = data
        
        batch_input_tokens = batch_input_tokens.cuda()
        batch_input_attentions = batch_input_attentions.cuda()
        batch_input_labels = batch_input_labels.cuda()
        
        """Prediction"""
        outputs = fine_model(batch_input_tokens, batch_input_attentions)
        loss_val = CELoss(outputs, batch_input_labels)
        
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(fine_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    fine_model.eval()
    p1 = CalP1(fine_model, dev_dataloader)
    print(f"Epoch: {epoch}번째 모델 성능(p@1): {p1}")
    if p1 > best_p1:
        best_p1 = p1
        print(f"BEST 성능(p@1): {best_p1}")
        SaveModel(fine_model, '.')