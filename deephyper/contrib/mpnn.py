"""This module provides practical functions to prepare a dataset for message passing neural networks.
"""
import numpy as np


def get_gcn_attention(edge_pair):
    """Calculate gcn attention coef for an edge pair.

    Args:
        edge_pair (array): # edges * 2
    """
    # The inverse of sqrt of node degrees
    source_node = np.unique(edge_pair[:, 0], return_counts=True, return_inverse=True)
    target_node = np.unique(edge_pair[:, 1], return_counts=True, return_inverse=True)
    source_neighbor = []
    for element in source_node[1]:
        source_neighbor.append(source_node[2][element])
    target_neighbor = []
    for element in target_node[1]:
        target_neighbor.append(target_node[2][element])
    # the number of neighboring nodes of each node in each edge pair
    source_neighbor = np.array(source_neighbor)
    target_neighbor = np.array(target_neighbor)

    neighbor_multiplication = np.multiply(source_neighbor, target_neighbor)
    gcn_attention = 1 / np.sqrt(neighbor_multiplication)
    return gcn_attention


def get_mol_feature(mol, max_node, max_edge):
    """Get molecular features for an rdkit molecule

    Args:
        mol (rdkit molecule)
        max_node (int): maximum number of nodes in a dataset
        max_edge (int): maximum number of edges in a dataset
    """
    node_feat = mol.nodes
    edge_pair = mol.pair_edges.T
    edge_feat = mol.pairs

    num_node = node_feat.shape[0]
    num_node_feat = node_feat.shape[1]
    num_edge = edge_feat.shape[0]
    num_edge_feat = edge_feat.shape[1]

    node_feat_new = np.zeros(shape=(max_node, num_node_feat))
    node_feat_new[:num_node, :num_node_feat] = node_feat

    edge_pair_new = np.ones(shape=(max_edge + max_node, 2)) * (max_node - 1)
    edge_pair_new[:num_edge, :] = edge_pair

    edge_pair_new[max_edge:, 0] = np.arange(max_node)
    edge_pair_new[max_edge:, 1] = np.arange(max_node)

    edge_feat_new = np.zeros(shape=(max_edge + max_node, num_edge_feat))
    edge_feat_new[:num_edge, :] = edge_feat

    mask = np.zeros(shape=(max_node,))
    mask[:num_node] = 1

    gcn_attention = get_gcn_attention(edge_pair_new)
    return node_feat_new, edge_pair_new, edge_feat_new, mask, gcn_attention


def get_all_mol_feat(data, max_node, max_edge):
    """Get molecular features for a whole dataset

    Args:
        data (weave data)
        max_node (int): maximum number of nodes in a dataset
        max_edge (int): maximum number of edges in a dataset
    """
    x = data.X
    y = data.y

    node_feat, edge_pair, edge_feat, mask, gcn_attention = [], [], [], [], []

    for mol in x:
        (
            node_feat_temp,
            edge_pair_temp,
            edge_feat_temp,
            mask_temp,
            gcn_attention_temp,
        ) = get_mol_feature(mol, max_node, max_edge)
        node_feat.append(node_feat_temp)
        edge_pair.append(edge_pair_temp)
        edge_feat.append(edge_feat_temp)
        mask.append(mask_temp)
        gcn_attention.append(gcn_attention_temp)

    if len(y.shape) == 1:
        y = y[..., None]

    return [
        np.array(node_feat),
        np.array(edge_pair),
        np.array(edge_feat),
        np.array(mask),
        np.array(gcn_attention),
    ], y
