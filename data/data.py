from collections import defaultdict


class Graph():
  def __init__(self):
    super(Graph, self).__init__()
    '''
        node_forward and node_backward are only used when building the data.
        Afterwards will be transformed into node_feature by DataFrame

        node_forward: name -> node_id
        node_backward: node_id -> feature_dict
        node_feature: a DataFrame containing all features
    '''
    self.node_forward = defaultdict(lambda: {})
    self.node_backward = defaultdict(lambda: [])
    self.node_feature = defaultdict(lambda: [])

    '''
        edge_list: index the adjacancy matrix (time) by
        <target_type, source_type, relation_type, target_id, source_id>
    '''
    self.edge_list = defaultdict(lambda: defaultdict(list))
    self.times = {}

  def add_node(self, node):
    nfl = self.node_forward[node['type']]
    if node['id'] not in nfl:
      self.node_backward[node['type']] += [node]
      ser = len(nfl)
      nfl[node['id']] = ser
      return ser
    return nfl[node['id']]

  def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
    edge = [self.add_node(source_node), self.add_node(target_node)]
    '''
        Add bi-directional edges with different relation type
    '''
    source_type = source_node['type']
    target_type = target_node['type']

    if source_type == 'paper' and target_type == 'fos':
        L = target_node['L']
        edge_type = f'{source_type}2{target_type}_{L}'
        rev_edge_type = f'{target_type}2{source_type}_{L}'
    else:
        edge_type = f'{source_type}2{target_type}'
        rev_edge_type = f'{target_type}2{source_type}'

    self.edge_list[edge_type][edge[1]].append(edge[0])
    self.edge_list[rev_edge_type][edge[0]].append(edge[1])
    self.times[time] = True

  def update_node(self, node):
    nbl = self.node_backward[node['type']]
    ser = self.add_node(node)
    for k in node:
      if k not in nbl[ser]:
        nbl[ser][k] = node[k]

  def get_meta_graph(self):
    types = self.get_types()
    metas = []
    for target_type in self.edge_list:
      for source_type in self.edge_list[target_type]:
        for r_type in self.edge_list[target_type][source_type]:
          metas += [(target_type, source_type, r_type)]
    return metas

  def get_types(self):
    return list(self.node_feature.keys())

