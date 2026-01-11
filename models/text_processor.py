'''
语言处理模块
用来加载bert模型
为语料库增加特殊token
为文本替换bbox token
提供prompt encoder
'''

import torch
import transformers
import torch.nn as nn
import os


class LanguageProcessor(nn.Module):
    def __init__(self, bbox_decoder=None, bert_type="bert-base-uncased", max_token = 60):
        super().__init__()
        self.tokenizer, self.text_encoder = self.load_bert_model(bert_type, max_token)
        self.bbox_decoder = bbox_decoder
        self.max_token = max_token


    def load_bert_model(self,bert_type="bert-base-uncased", max_token = 20):
        # 1. 定义一个本地存放模型的路径
        # 例如存放在当前目录下的 'saved_models/bert-base-uncased'
        local_path = os.path.join("/home/lhr/dataset/checkpoints/swin-unetr", bert_type)
        
        # 2. 检查本地路径是否存在
        if os.path.exists(local_path):
            print(f"Loading bert model and tokenizer from LOCAL path: {local_path}")
            # 从本地路径加载
            tokenizer = transformers.BertTokenizer.from_pretrained(local_path)
            text_encoder = transformers.BertModel.from_pretrained(local_path)
        else:
            print(f"Model not found locally. Downloading {bert_type} from HuggingFace...")
            # 从网络下载加载
            tokenizer = transformers.BertTokenizer.from_pretrained(bert_type)
            text_encoder = transformers.BertModel.from_pretrained(bert_type)
            
            # 3. 下载完后，保存到本地路径，供下次使用
            print(f"Saving model to {local_path}...")
            tokenizer.save_pretrained(local_path)
            text_encoder.save_pretrained(local_path)
        text_encoder.pooler = None
        return tokenizer, text_encoder

    def add_special_tokens(self, special_tokens: list):
        '''
        为tokenizer和text_encoder添加特殊token
        special_tokens: 特殊token列表
        '''
        add = 'additional_special_tokens'
        # 对应的特殊token指定id
        self.tokenizer.add_special_tokens({add: special_tokens})    
        # 查看特殊token的id
        self.special_tokens_ids = self.tokenizer.get_added_vocab()[special_tokens[0]]
        print(self.tokenizer.get_added_vocab())
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_embed_q(self, input_ids, bbox):
        '''
        为query添加bbox token
        input_ids: 输入的token id (B,l)
        bbox: 边界框坐标 (B, 4)
        '''
        # 处理bbox，将其变成对应的bbox_embeding
        if self.bbox_decoder is None:
            return input_ids, None
            # 返回的是元组，代表的是文本原来的token索引，文本现在的编码
        new_input_embeds = []
        if isinstance(bbox, (list, tuple)):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox_embeds = self.bbox_decoder(bbox) # (B, 1, C)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_bbox = (cur_input_ids == self.special_tokens_ids[0]).sum()
            bbox_token_indices = [0] + torch.where(cur_input_ids == self.special_tokens_ids[0]).tolist() + [-1]
            cur_input_ids_nobbox = []
            cur_bbox_idx = 0
            for i in range(len(bbox_token_indices) - 1):
                cur_input_ids_nobbox.append(cur_input_ids[bbox_token_indices[i]+1:bbox_token_indices[i+1]])
            cur_input_embeds = self.text_encoder(torch.cat(cur_input_ids_nobbox, dim=0)) # (B,L-count,C)
            split_sizes = [len(ids) for ids in cur_input_ids_nobbox]
            cur_input_embeds_nobbox = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            for i in range(num_bbox + 1):
                cur_new_input_embeds.append(cur_input_embeds_nobbox[i])
                if i < num_bbox:
                    cur_new_input_embeds.append(bbox_embeds[cur_bbox_idx])
                    cur_bbox_idx += 1
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            new_input_embeds.append(cur_new_input_embeds)
        # 现在new_input_embed是一个有bbox编码的编码，为(b,l,c)
        new_input_embeds = [x[:self.max_token] for x in new_input_embeds]
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))# 右填充
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        return None, new_input_embeds # 返回的是原来的token索引，和新的编码
    
    def forward(self, input_ids):
        return self.get_language_embeds(input_ids)

    def get_language_embeds(self, input_ids):
        '''
        获取语言编码
        input_ids: 输入的token id (B,l)
        return: 语言编码 (B,l,768)
        '''
        # 先检测末尾的 0 token
        # input_ids = input_ids[:, :torch.where(input_ids == 0)[1].max() + 1]
        # 将其作为attention_mask
        attention_mask = (input_ids != 0).long()
        return self.text_encoder(input_ids, attention_mask=attention_mask)['last_hidden_state']
    

def main():
    # 测试文本token功能与对应token的编码功能  
    lang_processor = LanguageProcessor()
    lang_processor.add_special_tokens(["[BBOX]"])
    text = "这是一个测试句子，包含[BBOX]特殊token。"
    tokens = lang_processor.tokenizer.tokenize(text)
    print(tokens)
    token_ids = lang_processor.tokenizer.convert_tokens_to_ids(tokens)
    # 将token_ids填充到max_len
    token_ids = token_ids + [0] * (lang_processor.max_len - len(token_ids))
    print(token_ids)
    lang_processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[BBOX]"]})
    # lang_processor.text_encoder.resize_token_embeddings(len(lang_processor.tokenizer))
    # print(lang_processor.tokenizer.get_added_vocab())
    # print(lang_processor.tokenizer.get_special_tokens_map())
    # print(lang_processor.tokenizer.get_vocab())
    # print(lang_processor.tokenizer.convert_ids_to_tokens(token_ids))
    # print(lang_processor.tokenizer.convert_tokens_to_ids(tokens))
    # print(lang_processor.tokenizer.convert_tokens_to_ids(["[BBOX]"]))
    print(lang_processor.get_language_embeds(torch.tensor([token_ids]))['last_hidden_state'].shape)
    '''
    {'[PAD]': 0, '[UNK]': 100, '[CLS]': 101, '[SEP]': 102, '[MASK]': 103, '[BBOX]': 30522}
    ['[UNK]', '[UNK]', '一', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '子', '，', '[UNK]', '[UNK]', '[BBOX]', '[UNK]', '[UNK]', 'token', '。']
    [100, 100, 1740, 100, 100, 100, 100, 1816, 1989, 100, 100, 30522, 100, 100, 19204, 1636]
    torch.Size([1, 16, 768])
    '''
if __name__ == '__main__':
    main()


            

