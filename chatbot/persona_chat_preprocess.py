import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer():
    def __init__(self, model_name:str='alaggung/bart-r3f', max_length:int=64, device=torch.device('cuda')) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side='left'
        self.max_length = max_length
        self.device = device
        self.model.eval().to(device)
    
    @torch.no_grad()
    def summarize(self, dialogues:list[list[str]]) -> str:
        inputs = self.tokenizer(
            ["[BOS]" + "[SEP]".join(dialogue) + "[EOS]" for dialogue in dialogues], 
            padding=True, 
            return_tensors='pt', 
            truncation=True,
            max_length=512
        ).to(self.device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        gen_ids = self.model.generate(**inputs, max_length=self.max_length, use_cache=True)
        
        summarized = [self.tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in gen_ids]
        return summarized
    
summarizer = Summarizer()

import json

INIT_UTTERS = 2
MAX_BATCH = 20

class Chat():
    def __init__(self, path) -> None:
        self.users = {}
        self.utterances = []
        self.utter_idx = INIT_UTTERS
        self.num_utters = 0
        self.__parse__(path)
    
    def __parse__(self, path):
        with open(path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
            
            # get persona
            info = data['info']
            personas = info['personas']
            
            for persona in personas:
                profiles = []
                for elem in persona['persona']:
                    profiles.append(elem['profile'])
                    
                self.users[persona['persona_id']] = ' '.join(profiles)
            
            # get utterances
            dialogues = []
            utterances = data['utterances']
            for utter in data['utterances']:
                del utter['utterance_id']
                dialogues.append(utter['text'])
            
            self.num_utters = len(utterances)
            
            # set utterance history
            
            numbers = list(range(self.num_utters))
            summarized = []
            target_index = [numbers[i:i+MAX_BATCH] for i in range(INIT_UTTERS, self.num_utters, MAX_BATCH)]
            for batch_target in target_index:
                summarized.extend(summarizer.summarize([dialogues[:i] for i in batch_target]))
            
            for i in range(INIT_UTTERS, self.num_utters):
                utterances[i]['history'] = summarized[i-INIT_UTTERS]
            
            self.utterances = utterances
            
    def next(self):
        if self.utter_idx >= len(self.utterances):
            return None
        
        cur_utter = self.utterances[self.utter_idx]
        role = self.users[cur_utter['persona_id']]
        label = cur_utter['text']
        history = cur_utter['history']
        query = self.utterances[self.utter_idx-1]['text']
        self.utter_idx += 1
        
        return {
            'role': role,
            'query': query,
            'history': history,
            'label': label
        }
        
    def reset(self):
        self.utter_idx = INIT_UTTERS
        
    def export(self, path):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump({
                'users': self.users,
                'utterances': self.utterances,
                'num_utters': self.num_utters
            }, file, ensure_ascii=False, indent=4)
        
import os
import glob
from tqdm import tqdm

def extract_chat(path):
    json_files = glob.glob(path)
    
    chat_list = []
    for i in tqdm(range(len(json_files))):
        file = json_files[i]
        chat = Chat(file)
        chat_list.append(chat)
        chat.export(os.path.join(f'data/train_data/{i}.json'))
        
    return chat_list
    
chat_list = extract_chat(os.path.join('data','044_persona_chat','train','*.json'))