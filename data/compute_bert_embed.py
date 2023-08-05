import dill
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, XLNetTokenizer
from transformers import BertModel, BertConfig, XLNetModel
import torch
import torch.nn as nn

domain = 'CS'

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, batch_size, max_seq_len):
        bert_model = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.df = df
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return index

    def collate(self, items):
        text_list = self.df.loc[items].tolist()
        if len(text_list) < self.batch_size:
            padd = []
            for i in range(self.batch_size - len(text_list)):
                padd.append('padding')
            text_list.extend(padd)
        tokens = self.tokenizer(text_list, max_length=self.max_seq_len,
                       truncation=True, padding='max_length', return_tensors='pt')
        return tokens

def compute_bert_embed(domain, node_type, feat_type, max_seq_length):
    data_dir = f'mag_small/graph_{domain}.pk'
    graph = dill.Unpickler(open(data_dir, 'rb')).load()
    df = graph.node_feature[node_type]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset(df[feat_type], 256, max_seq_length)
    params = {'batch_size': 256,
              'num_workers': 20,
              'prefetch_factor': 2,
              'collate_fn': dataset.collate,
              'shuffle': False,
              'drop_last': False}
    test_loader = torch.utils.data.DataLoader(dataset, **params)

    bert_model = 'bert-base-uncased'
    config = BertConfig.from_pretrained(bert_model)
    lm_model = BertModel.from_pretrained(bert_model, config=config)
    lm_model = nn.DataParallel(lm_model).to(device)

    lm_model.eval()
    feats = []
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as t_test_loader:
            for batch in t_test_loader:
                batch = batch.to(device)
                outputs = lm_model(**batch)
                feats.append(outputs['pooler_output'].cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    df['feat'] = list(feats)[:df.shape[0]]
    pq.write_table(pa.Table.from_pandas(df), f'mag_small/{domain}_{node_type}_bert.parquet')


def count_token_length(domain, node_type, feature_type):
    data_dir = f'output/graph_{domain}.pk'
    graph = dill.Unpickler(open(data_dir, 'rb')).load()
    text_list = graph.node_feature[node_type][feature_type].tolist()

    bert_model = "xlnet-base-cased"
    tokenizer = XLNetTokenizer.from_pretrained(bert_model)

    count = {4:0, 8:0, 16:0, 32:0, 64:0, 128:0, 256:0, 512:0}
    for text in text_list:
        tokens = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        num_tokens = torch.count_nonzero(tokens['attention_mask'])
        if num_tokens <= 4:
            count[4] += 1
        elif num_tokens <= 8:
            count[8] += 1
        elif num_tokens <= 16:
            count[16] += 1
        elif num_tokens <= 32:
            count[32] += 1
        elif num_tokens <= 64:
            count[64] += 1
        elif num_tokens <= 128:
            count[128] += 1
        elif num_tokens <= 256:
            count[256] += 1
        else:
            count[512] += 1

    print(count)


if __name__ == "__main__":
    compute_bert_embed(domain, 'paper', 'title', 128)
    compute_bert_embed(domain, 'fos', 'name', 64)
    compute_bert_embed(domain, 'venue', 'name', 64)

    #count_token_length(domain, 'paper', 'title')
    #count_token_length(domain, 'fos', 'name')
    #count_token_length(domain, 'venue', 'name')
