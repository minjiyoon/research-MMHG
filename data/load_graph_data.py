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
        self.feat_dims = {}
        self.relations = defaultdict(set)

        self.step_num = args.sample_depth
        self.sample_num = args.sample_num
        self.position_type = args.position_type

        data_dir = f'data/mag_small/graph_{args.dataset_domain}.pk'
        graph = dill.Unpickler(open(data_dir, 'rb')).load()

        max_node_num = -1
        for node_type in self.node_types:
            if node_type == 'paper':
                df = graph.node_feature[node_type].loc[:, [self.text_columns[node_type]]]
                df = df.rename(columns = {self.text_columns[node_type] : 'text'})
                df['id'] = list(range(df.shape[0]))
                self.df_target = df
            data_file = f'data/mag_small/{args.dataset_domain}_{node_type}_bert.parquet'
            df = pq.read_table(data_file).to_pandas()
            self.df_nodes[node_type] = np.asarray(list(df['feat']), dtype=np.float64)
            self.feat_dims[node_type] = df['feat'][0].shape[-1]
            if max_node_num < df['feat'].shape[0]:
                max_node_num = df['feat'].shape[0]

        # TEMPORARY: SHOULD BE UPDATED WITH MULTI-MODALITY
        self.hidden_size = self.feat_dims['paper']
        self.pad_id = max_node_num + 1

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

        self.config_computation_graph()

        self.label_type = args.label_type
        self.labels = graph.edge_list[f'fos2paper_{self.label_type}']
        self.label_set = list(graph.edge_list[f'paper2fos_{self.label_type}'].keys())
        self.label_num = len(self.label_set)

    def load_dataset(self):
        return Dataset.from_pandas(self.df_target)

    def config_computation_graph(self):
        target_type_list = ['paper']
        total_type_list = ['paper']
        num_neighbors = 1

        position_dict = {}
        if self.position_type == 'node_type':
            position_dict['paper'] = 0
        elif self.position_type == 'layer':
            position_dict[0] = 0
        elif self.position_type == 'layer_node_type':
            for layer in range(self.step_num + 1):
                position_dict[layer] = {}
            position_dict[0]['paper'] = 0
        position_ids = [0]
        curr_position_id = 1

        for layer in range(1, self.step_num + 1):
            new_target_type_list = []
            for target_type in target_type_list:
                for source_type in self.relations[target_type]:
                    new_target_type_list.extend(self.sample_num * [source_type])
                    total_type_list.extend(self.sample_num * [source_type])
                    num_neighbors += self.sample_num

                    if self.position_type == 'node_type':
                        if source_type not in position_dict.keys():
                            position_dict[source_type] = curr_position_id
                            curr_position_id += 1
                        position_ids.extend(self.sample_num * [position_dict[source_type]])
                    elif self.position_type == 'layer':
                        if layer not in position_dict.keys():
                            position_dict[layer] = curr_position_id
                            curr_position_id += 1
                        position_ids.extend(self.sample_num * [position_dict[layer]])
                    elif self.position_type == 'layer_node_type':
                        if source_type not in position_dict[layer].keys():
                            position_dict[layer][source_type] = curr_position_id
                            curr_position_id += 1
                        position_ids.extend(self.sample_num * [position_dict[layer][source_type]])
                    elif self.position_type == 'metapath':
                        position_ids.extend(self.sample_num * [curr_position_id])
                        curr_position_id += 1

            target_type_list = new_target_type_list

        if self.position_type == 'indiv':
            position_ids = list(range(num_neighbors))

        self.max_neighbors = num_neighbors
        self.position_ids = position_ids

        print("Maximum number of neighbors: ", num_neighbors)
        print("Total type list: ", total_type_list)
        print("Position type: ", self.position_type, ", position ids: ", position_ids)

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

    def sample_dup_computation_graph(self, seed_id):
        sampled_nodes = []
        sampled_nodes.append(('paper', seed_id))
        target_node_list = []
        target_node_list.append(('paper', seed_id))
        for layer in range(self.step_num):
            new_target_node_list = []
            for target_node in target_node_list:
                target_type = target_node[0]
                target_id = target_node[1]
                for source_type in self.relations[target_type]:
                    if target_id == self.pad_id:
                        source_ids = []
                    else:
                        source_ids = self.sample_neighbors(target_type, source_type, target_id)

                    if len(source_ids) == 0:
                        sampled_ids = self.sample_num * [self.pad_id]
                    elif len(source_ids) < self.sample_num:
                        sampled_ids = source_ids + (self.sample_num - len(source_ids)) * [self.pad_id]
                    else:
                        sampled_ids = np.random.choice(source_ids, self.sample_num, replace = False)

                    sampled_node_list = [(source_type, sampled_id) for sampled_id in sampled_ids]
                    sampled_nodes.extend(sampled_node_list)
                    new_target_node_list.extend(sampled_node_list)

            target_node_list = new_target_node_list

        # Convert to torch object
        sampled_feats = []
        sampled_attention_mask = []
        for node in sampled_nodes:
            node_type = node[0]
            node_id = node[1]
            if node_id == self.pad_id:
                sampled_feats.append(torch.zeros(self.feat_dims[node_type], dtype=torch.float))
                sampled_attention_mask.append(0)
            else:
                sampled_feats.append(torch.FloatTensor(self.df_nodes[node_type][node_id]))
                sampled_attention_mask.append(1)
        sampled_feats = torch.stack(sampled_feats, dim=0)
        sampled_attention_mask = torch.LongTensor(sampled_attention_mask)

        assert sampled_feats.shape[0] == self.max_neighbors
        return {'feats': sampled_feats, 'attention_mask': sampled_attention_mask}


    def get_label(self, seed_id):
        # Normalize label
        label_list = [self.label_set.index(fos) for fos in self.labels[seed_id]]
        label = torch.zeros(self.label_num, dtype=torch.float32)
        label[label_list] = 1

        return label


