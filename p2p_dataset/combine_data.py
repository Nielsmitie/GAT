import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
from dgl.graph import DGLGraph


'''
I can't find the exact data anywhere and no additional reference is given by the author.
Some datasets are available but all have different formats.

The authors refuses to upload more to the ppi experiments. So the process_ppi function has to be used.

The data from https://docs.dgl.ai/en/latest/_modules/dgl/data/ppi.html sounds really similar to the one used in the paper.
Unfortunately the data looks different than the one expected from process_ppi.

The code from process_ppi is so unreadable that I have no other choice than to conform to the input format.
'''

# code from https://docs.dgl.ai/en/latest/_modules/dgl/data/ppi.html

with open('../p2p_dataset/train_graph.json'.format(dir)) as jsonfile:
    g_data_1 = json.load(jsonfile)
labels_1 = np.load('../p2p_dataset/train_labels.npy'.format(dir))
features_1 = np.load('../p2p_dataset/train_feats.npy'.format(dir))
graph_id_1 = np.load('../p2p_dataset/train_graph_id.npy'.format(dir))

with open('../p2p_dataset/valid_graph.json'.format(dir)) as jsonfile:
    g_data_2 = json.load(jsonfile)
labels_2 = np.load('../p2p_dataset/valid_labels.npy'.format(dir))
features_2 = np.load('../p2p_dataset/valid_feats.npy'.format(dir))
graph_id_2 = np.load('../p2p_dataset/valid_graph_id.npy'.format(dir))

with open('../p2p_dataset/test_graph.json'.format(dir)) as jsonfile:
    g_data_3 = json.load(jsonfile)
labels_3 = np.load('../p2p_dataset/test_labels.npy'.format(dir))
features_3 = np.load('../p2p_dataset/test_feats.npy'.format(dir))
graph_id_3 = np.load('../p2p_dataset/test_graph_id.npy'.format(dir))

# combine all data into one file so it can be processed by process_ppi.py
g_data = g_data_1
g_data['nodes'].extend(g_data_2['nodes'])
g_data['links'].extend(g_data_2['links'])
g_data['nodes'].extend(g_data_3['nodes'])
g_data['links'].extend(g_data_3['links'])
labels = np.vstack([labels_1, labels_2, labels_3])
features = np.vstack([features_1, features_2, features_3])
graph_id = np.concatenate([graph_id_1, graph_id_2, graph_id_3])


np.save('../p2p_dataset/ppi-feats.npy', features)

# todo add information validation and test?!
with open('../p2p_dataset/ppi-G.json', 'w') as outfile:
    json.dump(g_data, outfile)

# todo missing convert to dict? Don't know which properties are expected
np_array_to_list = labels.tolist()
with open('../p2p_dataset/ppi-class_map.json', 'w') as outfile:
    json.dump(np_array_to_list, outfile, sort_keys=True, indent=4)

np_array_to_list = graph_id.tolist()
with open('../p2p_dataset/ppi-id_map.json', 'w') as outfile:
    json.dump(np_array_to_list, outfile, sort_keys=True, indent=4)
