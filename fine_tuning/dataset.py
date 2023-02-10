import json, pdb
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class fine_loader(Dataset):
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        special_tokens = {'sep_token': '<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        
        """ 세션 데이터 """
        with open(data_path, 'r') as f:
            self.session_dataset = json.load(f)
        
    def __len__(self): # 기본적인 구성
        return len(self.session_dataset)
    
    def __getitem__(self, idx): # 기본적인 구성
        session = self.session_dataset[str(idx)]
        context = session['context']
        positive_response = session['positive_response']
        negative_responses = session['negative_responses']
        session_tokens = []
        session_labels = []
        
        """ MRS 입력 """
        context_token = [self.tokenizer.cls_token_id]
        for utt in context:
            context_token += self.tokenizer.encode(utt, add_special_tokens=False)
            context_token += [self.tokenizer.sep_token_id]
        
        pos_respons_token = [self.tokenizer.eos_token_id]
        pos_respons_token += self.tokenizer.encode(positive_response, add_special_tokens=False)
        positive_tokens = context_token + pos_respons_token
        session_tokens.append(positive_tokens)
        session_labels.append(1)
        
        for negative_response in negative_responses:
            neg_respons_token = [self.tokenizer.eos_token_id]
            neg_respons_token += self.tokenizer.encode(negative_response, add_special_tokens=False)
            negative_tokens = context_token + neg_respons_token        
            session_tokens.append(negative_tokens)
            session_labels.append(0)
        
        return session_tokens, session_labels
    
    def collate_fn(self, sessions): # 배치를 위한 구성
        '''
            input:
                data: [(session1), (session2), ... ]
            return:
                batch_input_tokens_pad: (B, L) padded
                batch_labels: (B)
        '''        
        # 최대 길이 찾기 for padding
        max_input_len = 0
        for session in sessions:
            session_tokens, session_labels = session
            input_tokens_len = [len(x) for x in session_tokens]
            max_input_len = max(max_input_len, max(input_tokens_len))
        
        batch_input_tokens, batch_input_labels = [], []
        batch_input_attentions = []
        for session in sessions:
            session_tokens, session_labels = session
            for session_token in session_tokens:
                input_token = session_token + [self.tokenizer.pad_token_id for _ in range(max_input_len-len(session_token))]
                input_attention = [1 for _ in range(len(session_token))] + [0 for _ in range(max_input_len-len(session_token))]
                batch_input_tokens.append(input_token)
                batch_input_attentions.append(input_attention)
            batch_input_labels += session_labels
        
        return torch.tensor(batch_input_tokens), torch.tensor(batch_input_attentions), torch.tensor(batch_input_labels)