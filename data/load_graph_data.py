from time import perf_counter
import dill
import pickle
import pyarrow.parquet as pq
import random
import re
import os
from collections import defaultdict
from functools import partial

from scipy import sparse as sps
from sklearn.preprocessing import normalize
from datasets import Dataset
import numpy as np
import torch


class OAGDataset:
    def __init__(self, args):

        self.node_types = ['paper', 'venue', 'fos']
        self.text_columns = {'paper': 'title', 'venue': 'name', 'fos': 'name'}
        self.edge_types = ['paper2paper', 'paper2fos', 'paper2venue', 'fos2paper', 'venue2paper']

        self.df_nodes = {}
        self.df_edges = {}
        self.relations = defaultdict(set)

        self.step_num = args.sample_depth
        self.sample_num = args.sample_num

        data_dir = f'data/mag_small/graph_{args.dataset_domain}.pk'
        graph = dill.Unpickler(open(data_dir, 'rb')).load()

        for node_type in self.node_types:
            if node_type == 'paper':
                df = graph.node_feature[node_type].loc[:, [self.text_columns[node_type]]]
                df = df.rename(columns = {self.text_columns[node_type] : 'text'})
                df['id'] = list(range(df.shape[0]))
                self.df_target = df
            data_file = f'data/mag_small/{args.dataset_domain}_{node_type}_bert.parquet'
            df = pq.read_table(data_file).to_pandas()
            self.df_nodes[node_type] = np.asarray(list(df['feat']), dtype=np.float64)

        for edge_type in self.edge_types:
            src_type = edge_type[:edge_type.find('2')]
            trg_type = edge_type[edge_type.find('2') + 1:]
            self.relations[trg_type].update([src_type])
            if (src_type == 'paper' and trg_type == 'fos') or (src_type == 'fos' and trg_type == 'paper'):
                label_level = int(args.label_type[1])
                merged_dict = {}
                for L in range(label_level + 1, 6):
                    dict1 = graph.edge_list[f'{edge_type}_L{L}']
                    merged_dict = {**merged_dict, **dict1}
                self.df_edges[edge_type] = merged_dict
            else:
                self.df_edges[edge_type] = graph.edge_list[edge_type]

        self.max_neighbors = self.compute_max_neighbors()

        self.label_type = args.label_type
        self.labels = graph.edge_list[f'fos2paper_{self.label_type}']
        self.label_set = list(graph.edge_list[f'paper2fos_{self.label_type}'].keys())
        self.label_num = len(self.label_set)

    def load_dataset(self):
        return Dataset.from_pandas(self.df_target)

    def compute_max_neighbors(self):
        num_neighbors = 1
        target_type_list = ['paper']
        for _ in range(self.step_num):
            new_target_type_list = []
            for target_type in target_type_list:
                for source_type in self.relations[target_type]:
                    new_target_type_list.extend(self.sample_num * [source_type])
                    num_neighbors += self.sample_num
            target_type_list = new_target_type_list
        return num_neighbors

    def sample_neighbors(self, target_type, source_type, target_id):
        relation_type = f'{source_type}2{target_type}'
        if target_type == 'paper' and source_type == 'paper':
            target_type = 'dst_paper'
            source_type = 'src_paper'
        if target_id in self.df_edges[relation_type].keys():
            source_ids = self.df_edges[relation_type][target_id]
        else:
            source_ids = []
        return source_ids

    def sample_computation_graph(self, seed_id):
        sampled_nodes = defaultdict(list)
        sampled_nodes['paper'].append(seed_id)

        target_type_list = set(['paper'])
        target_node_list = defaultdict(list)
        target_node_list['paper'].append(seed_id)
        for _ in range(self.step_num):
            new_target_type_list = set()
            new_target_node_list = defaultdict(list)
            for target_type in target_type_list:
                for source_type in self.relations[target_type]:
                    new_target_type_list.add(source_type)
                    for target_id in target_node_list[target_type]:
                        source_ids = self.sample_neighbors(target_type, source_type, target_id)
                        if len(source_ids) == 0:
                            continue
                        elif len(source_ids) < self.sample_num:
                            sampled_ids = source_ids
                        else:
                            sampled_ids = np.random.choice(source_ids, self.sample_num, replace = False)
                        sampled_nodes[source_type].extend(sampled_ids)
                        new_target_node_list[source_type].extend(sampled_ids)
            target_type_list = new_target_type_list
            target_node_list = new_target_node_list

        # Convert to torch object
        sampled_feats = []
        for node_type in sampled_nodes.keys():
            sampled_ids = sampled_nodes[node_type]
            sampled_feats.append(torch.FloatTensor(self.df_nodes[node_type][sampled_ids]))
        sampled_feats = torch.cat(sampled_feats, dim=0)
        sampled_attention_mask = torch.ones((sampled_feats.shape[0]))

        # Normalize feature
        feats = torch.zeros((self.max_neighbors, sampled_feats.shape[1]))
        feats[:sampled_feats.shape[0]] = sampled_feats
        attention_mask = torch.zeros((self.max_neighbors))
        attention_mask[:sampled_attention_mask.shape[0]] = sampled_attention_mask

        return {'feats': feats, 'attention_mask': attention_mask}


    def get_label(self, seed_id):
        # Normalize label
        label_list = [self.label_set.index(fos) for fos in self.labels[seed_id]]
        label = torch.zeros(self.label_num, dtype=torch.float32)
        label[label_list] = 1

        return label


