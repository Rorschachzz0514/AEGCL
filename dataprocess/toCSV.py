import scipy.io as scio
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import numpy as np
import pandas as pd

dataset_name = "acmv9"
path = "../data/acmv9.mat"
# path = "../Citationv1/citationv1.mat"
# path = "../Dblpv7/dblpv7.mat"
def matrix2coo(m):
    """
    Change normal matrix to coo sparse matrix

    Input:
        m:      (np.array 2dim) input normal matrix
    Output:
        coo_m:  (sp.coo_matrix) output coo sparse matrix
    """
    rows, cols, values = [], [], []
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            if m[i,j] != 0:
                rows.append(i)
                cols.append(j)
                values.append(m[i,j])
    coo_m = sp.coo_matrix((values, (rows, cols)), shape = m.shape, dtype = float)

    return coo_m

if __name__ == "__main__":
    data = scio.loadmat(path)
    x = data['attrb']
    label = data['group']
    is_mul_label = np.sum(label,1)>1
    num_pre = 0
    index_dict = {}
    graph = nx.from_numpy_matrix(data['network'])
    j = 0
    for i in range(x.shape[0]):
        # 删除孤立节点
        # 删除多标签节点
        tmp = set(graph.neighbors(i))
        if i in tmp:
            tmp.remove(i)
        if is_mul_label[i] or len(tmp) == 0:
            graph.remove_node(i)
        else:
            index_dict[i] = j
            j += 1
    node_nums = graph.number_of_nodes()
    x_new = np.zeros((node_nums, x.shape[1]))
    label_new = np.zeros((node_nums, label.shape[1]))
    for node in graph.nodes():
        x_new[index_dict[node]] = x[node]
        label_new[index_dict[node]] = label[node]

    graph_dict = {}
    for nodeindex in graph.nodes():
        graph_dict[index_dict[nodeindex]] = [index_dict[nb] for nb in list(graph.neighbors(nodeindex)) if not is_mul_label[nb] and nodeindex != nb]
    #对原始数据进行shuffle
    # permutation_index = np.arange(len(label_new))
    # np.random.shuffle(permutation_index)
    # x_new[permutation_index, :] = x_new[np.arange(len(label_new)), :]
    # label_new[permutation_index, :] = label_new[np.arange(len(label_new)), :]
    # graph_dict_shuffle = dict()
    # for i in range(len(label_new)):
    #     graph_dict_shuffle[permutation_index[i]] = [permutation_index[nb] for nb in graph_dict[i]]

    with open(f"../data/ind.{dataset_name}.allx", "wb") as f:
        pkl.dump(matrix2coo(x_new), f)
    with open(f"../data/ind.{dataset_name}.ally", "wb") as f:
        pkl.dump(label_new, f)
    with open(f"../data/ind.{dataset_name}.graph", "wb") as f:
        pkl.dump(graph_dict, f)