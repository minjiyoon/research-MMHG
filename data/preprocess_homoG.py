import dill
import numpy as np
import random
from collections import defaultdict
import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset
from transformers import BertTokenizer, XLNetTokenizer
from transformers import BertModel, BertConfig, XLNetModel
import torch
import torch.nn as nn




def preprocess_cora():
    # Node labels and original ids
    path = 'RAW/cora/cora'
    idx_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
    data_citeid = idx_features_labels[:, 0].tolist()
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = [class_map[l] for l in labels]

    df = pd.DataFrame({
        'org_id': data_citeid,
        'label': data_Y,
    })

    # Edge list
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.dtype(int)).reshape(edges_unordered.shape)

    edge_list = defaultdict(list)
    for edge in edges:
        if edge[0] is not None and edge[1] is not None:
            edge_list[edge[0]].append(edge[1])
            edge_list[edge[1]].append(edge[0])

    # Text information on nodes
    with open('RAW/cora/mccallum/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'RAW/cora/mccallum/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)

    # Save node/edge information
    df['text'] = text
    pq.write_table(pa.Table.from_pandas(df), f'cora/cora_preprocess.parquet')
    with open(f'cora/cora_edges.pkl', 'wb') as file:
        dill.dump(edge_list, file)


def preprocess_pubmed():
    path = 'RAW/pubmed/'

    n_nodes = 19717
    n_features = 500

    data_Y = [None] * n_nodes
    data_edges = []
    data_pubid = [None] * n_nodes
    paper_to_index = {}

    # Node labels and original ids
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0
        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - 1  # subtract 1 to zero-count
            data_Y[i] = label

    df = pd.DataFrame({
        'org_id': data_pubid,
        'label': data_Y,
    })

    # Edge list
    edge_list = defaultdict(list)
    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):
            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')
            edge_id = items[0]
            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            if head != tail:
                edge_list[paper_to_index[head]].append(paper_to_index[tail])
                edge_list[paper_to_index[tail]].append(paper_to_index[head])

    # Text information on nodes
    f = open('RAW/pubmed/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)

    # Save node/edge information
    df['text'] = text
    pq.write_table(pa.Table.from_pandas(df), f'pubmed/pubmed_preprocess.parquet')
    with open(f'pubmed/pubmed_edges.pkl', 'wb') as file:
        dill.dump(edge_list, file)


def preprocess_ogbn_arxiv():
    # Node labels and edge list
    dataset = PygNodePropPredDataset(root='RAW/', name='ogbn-arxiv')
    idx_splits = dataset.get_idx_split()
    data = dataset[0]
    data_Y = data.y.tolist()

    edge_list = defaultdict(list)
    for edge in data.edge_index.t().tolist():
        edge_list[edge[0]].append(edge[1])
        edge_list[edge[1]].append(edge[0])

    # Text information on nodes
    nodeidx2paperid = pd.read_csv('RAW/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    raw_text = pd.read_csv('RAW/ogbn-arxiv-raw/titleabs.tsv', sep='\t', header=None, names=['paper id', 'title', 'abs'])
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)

    # Save node/edge information
    df = pd.DataFrame({
        'label': data_Y,
        'text': text
    })
    pq.write_table(pa.Table.from_pandas(df), f'ogbn-arxiv/ogbn-arxiv_preprocess.parquet')
    with open(f'ogbn-arxiv/ogn-arxiv_edges.pkl', 'wb') as file:
        dill.dump(edge_list, file)
        dill.dump(idx_splits, file)


def preprocess_GIANT_data(data_name):
    # Node labels and edge list
    dataset = PygNodePropPredDataset(root='RAW/', name=data_name)
    idx_splits = dataset.get_idx_split()
    data = dataset[0]
    data_Y = data.y.tolist()

    edge_list = defaultdict(list)
    for edge in data.edge_index.t().tolist():
        edge_list[edge[0]].append(edge[1])
        edge_list[edge[1]].append(edge[0

    # Text information on nodes
    with open(f'RAW/{data_name}-raw/X.all.txt', 'r') as fin:
        text = fin.readlines()

    # Save node/edge information
    df = pd.DataFrame({
        'label': data_Y,
        'text': text
    })
    pq.write_table(pa.Table.from_pandas(df), f'{data_name}/{data_name}_preprocess.parquet')
    with open(f'{data_name}/{data_name}_edges.pkl', 'wb') as file:
        dill.dump(edge_list, file)
        dill.dump(idx_splits, file)


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

def compute_bert_embed(dataset, max_seq_length):
    data_file = f'{dataset}/{dataset}_preprocess.parquet'
    df = pq.read_table(data_file).to_pandas()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset(df['text'], 256, max_seq_length)
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
    df['feat'] = list(feats)
    pq.write_table(pa.Table.from_pandas(df), f'{dataset}/{dataset}_bert.parquet')

if __name__ == "__main__":
    preprocess_cora()
    preprocess_pubmed()
    preprocess_ogbn_arxiv()

    #preprocess_GIANT_data('ogbn-arxiv')
    preprocess_GIANT_data('ogbn-product')

    compute_bert_embed('cora', 512)
    compute_bert_embed('pubmed', 512)
    compute_bert_embed('ogbn_arxiv', 512)
    compute_bert_embed('ogbn_product', 512)

