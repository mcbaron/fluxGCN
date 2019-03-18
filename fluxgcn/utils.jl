# utils.jl

using PyCall
using SparseArrays
using LinearAlgebra

const scipy_sparse_find = pyimport("scipy.sparse")["find"]
function jlsparse(Apy::PyObject)
    IA, JA, SA = scipy_sparse_find(Apy)
    return sparse(Int[i+1 for i in IA], Int[i+1 for i in JA], SA)
end

function jlload_data(dataset_str)
    # Loads input data from gcn/data directory
py"""
import sys
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask"""

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = py"load_data"(dataset_str)
    features = jlsparse(features)
    adj = jlsparse(adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
end

function normalize_adj(adj)
    # Symmetrically normalize adjacency matrix.
    rowsum = sum(adj, dims=2)
    d_inv_sqrt = vec(rowsum.^(-.5))
    if any(d_inv_sqrt .== Inf)
        d_inv_sqrt[isinf.(d_inv_sqrt)] = 0.0
    end
    d_mat_inv_sqrt = spdiagm(0 => d_inv_sqrt)
    return (adj * d_mat_inv_sqrt)' * d_mat_inv_sqrt
end

function preprocess_adj(adj)
    # Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    adj_normalized = normalize_adj(adj) + sparse(1*I, size(adj))
    return adj_normalized #sparse_to_tuple
end

function masked_accuracy(preds, labels, mask)
    # accuracy with masking
    correct_pred = equal(argmax(preds, 1), argmax(labels, 1))
    accuracy_all = cast(correct_pred, Float32)
    mask = cast(mask, dtype=Float32)
    mask /= reduce_mean(mask)
    accuracy_all *= mask
    return reduce_mean(accuracy_all)
end
