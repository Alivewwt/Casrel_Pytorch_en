import json
import os
import torch
import numpy as np
from random import choice
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

BERT_MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained('../bert_model/bert_base')

class CMEDDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer
        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        encoding = self.tokenizer(text,max_length = BERT_MAX_LEN, truncation=True)
        tokens = encoding.tokens()
        text_len = len(tokens)
        ins_json_data_f = [[triple[0][0],triple[1],triple[2][0]]for triple in ins_json_data['triple_list']]
        if not self.is_test:
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                sub_head_idx = encoding.char_to_token(triple[0][1])
                obj_head_idx = encoding.char_to_token(triple[2][1])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub_tail_idx = encoding.char_to_token(triple[0][2])
                    sub = (sub_head_idx, sub_tail_idx)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    obj_tail_idx = encoding.char_to_token(triple[2][2])
                    s2ro_map[sub].append((obj_head_idx, obj_tail_idx, self.rel2id[triple[1]]))

            if s2ro_map:
                token_ids,masks = encoding['input_ids'],encoding['attention_mask']
                token_ids = np.array(token_ids)
                masks = np.array(masks)
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data_f, tokens, text
            else:
                return None
        else:
            token_ids,masks = encoding['input_ids'],encoding['attention_mask']
            token_ids = np.array(token_ids)
            masks = np.array(masks)
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data_f, tokens, text


def cmed_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens, text = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_heads = torch.Tensor(cur_batch, max_text_len, 4).zero_()
    batch_obj_tails = torch.Tensor(cur_batch, max_text_len, 4).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
        batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
        batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'obj_heads': batch_obj_heads,
            'obj_tails': batch_obj_tails,
            'triples': triples,
            'tokens': tokens,
            'text': text}


def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=cmed_collate_fn):
    dataset = CMEDDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.Stream()  
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        # with torch.Stream(self.stream):
        for k, v in self.next_data.items():
            if isinstance(v, torch.Tensor):
                self.next_data[k] = self.next_data[k]

    def next(self):
        # torch.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
