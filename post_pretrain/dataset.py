import csv
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class post_loader(Dataset):
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        special_tokens = {'sep_token': '<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        
        f = open(data_path, 'r')
        rdr = csv.reader(f)        
        
        """ 세션 데이터 저장할 것"""
        session_dataset = []
        session = []

        """ 실제 데이터 저장 방식 """
        for i, line in enumerate(rdr):
            if i == 0:
                header  = line
            else:
                utt = line[0]
                if utt.strip() != '':
                    session.append(utt)
                else:
                    """ 세션 데이터 저장 """
                    session_dataset.append(session)
                    session = []
        """ 마지막 세션 저장 """
        session_dataset.append(session)
        f.close()
        
        """ short session context """
        k = 4 # 논문에서 가장 좋았던 숫자
        self.short_session_dataset = []
        for session in session_dataset:    
            for i in range(len(session)-k+1):
                self.short_session_dataset.append(session[i:i+k])
                
        """ 모든 발화 저장 """
        self.all_utts = set()
        for session in session_dataset:
            for utt in session:
                self.all_utts.add(utt)
        self.all_utts = list(self.all_utts)
        
    def __len__(self): # 기본적인 구성
        return len(self.short_session_dataset)
    
    def __getitem__(self, idx): # 기본적인 구성
        session = self.short_session_dataset[idx]
        """ MLM 입력 """
        mask_ratio = 0.15
        self.corrupt_tokens = []
        self.output_tokens = []
        for i, utt in enumerate(session):
            original_token = self.tokenizer.encode(utt, add_special_tokens=False)

            mask_num = int(len(original_token)*mask_ratio)
            mask_positions = random.sample([x for x in range(len(original_token))], mask_num)
            corrupt_token = []
            for pos in range(len(original_token)):
                if pos in mask_positions:
                    corrupt_token.append(self.tokenizer.mask_token_id)
                else:
                    corrupt_token.append(original_token[pos])

            if i == len(session)-1:
                self.output_tokens += original_token
                self.corrupt_tokens += corrupt_token
            else:
                self.output_tokens += original_token + [self.tokenizer.sep_token_id]
                self.corrupt_tokens += corrupt_token + [self.tokenizer.sep_token_id]    
        
        """ label for loss """
        self.corrupt_mask_positions = []
        for pos in range(len(self.corrupt_tokens)):
            if self.corrupt_tokens[pos] == self.tokenizer.mask_token_id:
                self.corrupt_mask_positions.append(pos)                
                
        """ URC 입력 """
        urc_tokens = []
        context_utts = []
        for i in range(len(session)):
            utt = session[i]    
            original_token = self.tokenizer.encode(utt, add_special_tokens=False)
            if i == len(session)-1:
                urc_tokens += [self.tokenizer.eos_token_id]
                """ 기존 response 입력 """
                self.positive_tokens = [self.tokenizer.cls_token_id] + urc_tokens + original_token
                """ random negative respons 입력 """
                while True:
                    random_neg_response = random.choice(self.all_utts)
                    if random_neg_response not in context_utts:
                        break
                random_neg_response_token = self.tokenizer.encode(random_neg_response, add_special_tokens=False)
                self.random_tokens = [self.tokenizer.cls_token_id] + urc_tokens + random_neg_response_token
                """ context negative response 입력 """
                context_neg_response = random.choice(context_utts)
                context_neg_response_token = self.tokenizer.encode(context_neg_response, add_special_tokens=False)
                self.context_neg_tokens = [self.tokenizer.cls_token_id] + urc_tokens + context_neg_response_token
            else:
                urc_tokens += original_token + [self.tokenizer.sep_token_id]
            context_utts.append(utt)
        
        return self.corrupt_tokens, self.output_tokens, self.corrupt_mask_positions, [self.positive_tokens, self.random_tokens, self.context_neg_tokens], [0, 1, 2]
    
    def collate_fn(self, sessions): # 배치를 위한 구성
        '''
            input:
                data: [(session1), (session2), ... ]
            return:
                batch_corrupt_tokens: (B, L) padded
                batch_output_tokens: (B, L) padded
                batch_corrupt_mask_positions: list
                batch_urc_inputs: (B, L) padded
                batch_urc_labels: (B)
                batch_mlm_attentions
                batch_urc_attentions

            batch가 3
            MLM = 3개의 입력데이터 (입력데이터별로 길이가 다름)
            URC = 9개의 입력데이터 (context는 길이가 다름, response candidate도 길이가 다름)
        '''
        batch_corrupt_tokens, batch_output_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_urc_labels = [], [], [], [], []
        batch_mlm_attentions, batch_urc_attentions = [], []
        # MLM, URC 입력에 대해서 가장 긴 입력 길이를 찾기
        corrupt_max_len, urc_max_len = 0, 0
        for session in sessions:
            corrupt_tokens, output_tokens, corrupt_mask_positions, urc_inputs, urc_labels = session
            if len(corrupt_tokens) > corrupt_max_len:
                corrupt_max_len = len(corrupt_tokens)
            positive_tokens, random_tokens, context_neg_tokens = urc_inputs
            if max([len(positive_tokens), len(random_tokens), len(context_neg_tokens)]) > urc_max_len:
                urc_max_len = max([len(positive_tokens), len(random_tokens), len(context_neg_tokens)])
                
        ## padding 토큰을 추가하는 부분
        for session in sessions:
            corrupt_tokens, output_tokens, corrupt_mask_positions, urc_inputs, urc_labels = session
            """ mlm 입력 """
            batch_corrupt_tokens.append(corrupt_tokens + [self.tokenizer.pad_token_id for _ in range(corrupt_max_len-len(corrupt_tokens))])
            batch_mlm_attentions.append([1 for _ in range(len(corrupt_tokens))] + [0 for _ in range(corrupt_max_len-len(corrupt_tokens))])
            
            """ mlm 출력 """
            batch_output_tokens.append(output_tokens + [self.tokenizer.pad_token_id for _ in range(corrupt_max_len-len(corrupt_tokens))])
            
            """ mlm 레이블 """
            batch_corrupt_mask_positions.append(corrupt_mask_positions)
            
            """ urc 입력 """
            # positive_tokens, random_tokens, context_neg_tokens = urc_inputs
            for urc_input in urc_inputs:                            
                batch_urc_inputs.append(urc_input + [self.tokenizer.pad_token_id for _ in range(urc_max_len-len(urc_input))])
                batch_urc_attentions.append([1 for _ in range(len(urc_input))] + [0 for _ in range(urc_max_len-len(urc_input))])
            
            """ urc 레이블 """
            batch_urc_labels += urc_labels
        return torch.tensor(batch_corrupt_tokens), torch.tensor(batch_output_tokens), batch_corrupt_mask_positions, torch.tensor(batch_urc_inputs), \
        torch.tensor(batch_urc_labels), torch.tensor(batch_mlm_attentions), torch.tensor(batch_urc_attentions)