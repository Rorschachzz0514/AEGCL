import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import time
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_utils import loadAllData
from model import Model
import numpy as np
import networkx as nx
from torch_sparse import coalesce
from sklearn.cluster import KMeans
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
import numpy as np
from sklearn import metrics

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr=None, p=0.5, lower=0.1, upper=1, force_undirected=False,
                num_nodes=None, training=True, is_adaptive=True):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training or p == 0.0:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)
    
    if is_adaptive:#根据边权重，计算drop概率
        probability = torch.where(torch.gt(edge_attr, 50), torch.zeros_like(edge_attr).fill_(50), edge_attr)
        a = lower
        b = upper
        k = (b - a) / (torch.max(edge_attr) - torch.min(edge_attr))
        probability = a + k * (edge_attr - torch.min(edge_attr))
        mask = torch.bernoulli(probability).to(torch.bool)
        #print("auto")
    else:
        mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
        mask = torch.bernoulli(mask).to(torch.bool)
        print("unauto")

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

# 动态计算mask概率
def drop_feature(x, prob):
    drop_mask = torch.empty(
        (x.size(0), x.size(1)),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < prob.reshape(prob.size()[0], 1)
    x = x.clone()
    x[drop_mask] = 0

    return x

# 固定概率mask
def drop_feature_origin(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def train(model: Model, x, motifs_all, motifs_num, edge_index, edge_weight, is_node_adaptive, is_motif,
          is_edge_weight, is_adaptive, node_probability, lower=0.1, upper=1):
    model.train()
    optimizer.zero_grad()
    is_adaptive=True
    edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, drop_edge_rate_1, lower, upper, is_adaptive=is_adaptive)
    edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, drop_edge_rate_2, lower, upper, is_adaptive=is_adaptive)
    if is_node_adaptive:
        x_1 = drop_feature(x, node_probability)
        x_2 = drop_feature(x, node_probability)
        print('node_adaptive')
    else:
        x_1 = drop_feature_origin(x, drop_feature_rate_1)
        x_2 = drop_feature_origin(x, drop_feature_rate_2)
    if is_edge_weight:
        z1 = model(x_1, edge_index_1, edge_weight_1, motifs_all, motifs_num)
        z2 = model(x_2, edge_index_2, edge_weight_2, motifs_all, motifs_num)
        print('edge_weight')

    else:
        z1 = model(x_1, edge_index_1, None, motifs_all, motifs_num)
        z2 = model(x_2, edge_index_2, None, motifs_all, motifs_num)
    
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_test(model, x, motifs_all, motifs_num, edge_index, edge_weight, y, is_edge_weight, is_motif, final=False):
    model.eval()
    # if is_motif:
    #     x = torch.cat([x, motifs_num.T], dim=1)
    if is_edge_weight:
        z = model(x, edge_index, edge_weight, motifs_all, motifs_num)
        #print('yes edge weight')
    else:
        z = model(x, edge_index, None, motifs_all, motifs_num)
        #print('no edge weight')
    ratio = 0.1
    #if args.dataset == 'email_motif':
    #    ratio = 0.6
    labels = y.cpu().numpy()
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=501,n_iter=3000000,perplexity=50)
    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        total = 0
        for i in range(len(ind[0])):
            total += w[ind[0][i], ind[1][i]]
        return total * 1.0 / Y_pred.size, w

    y_pred_labels = KMeans(n_clusters=len(set(labels))).fit_predict(z.detach().cpu().numpy())



    nmi = metrics.normalized_mutual_info_score(labels, y_pred_labels)
    #print("acc")
    #print(cluster_acc(labels, y_pred_labels))
    #print("nmi")
    #print(nmi)
    return label_classification(z, y, ratio=0.1),cluster_acc(y_pred_labels, labels),nmi


if __name__ == '__main__':
    import torch
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.cuda.device_count()

    # import os
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    #
    # torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='1')
    #parser.add_argument('--gpu_id', '-g', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--is_node_weight', '-n', action='store_true')
    parser.add_argument('--is_edge_weight', '-w', action='store_true')
    parser.add_argument('--is_adaptive', '-a', action='store_true')
    parser.add_argument('--is_motif', '-m', action='store_true')
    args = parser.parse_args()
    #assert args.gpu_id in range(0, 8)
    #torch.cuda.set_device(args.gpu_id)
    # is_node_weight = args.is_node_weight
    # is_edge_weight = args.is_edge_weight
    # is_adaptive = args.is_adaptive
    #is_motif = args.is_motif

    is_node_weight = True
    is_edge_weight = True
    is_adaptive = True
    is_motif = True
    # for datasets in [["acmv9", "Football"],["dblpv7", "Football"],["citationv1", "Football"],["amazon-photo", "Football"],["amazon-computers", "Football"]]:
    for tau in [0.4]:
        for datasets in [["acmv9", "Football"]]:
            config = yaml.load(open(args.config), Loader=SafeLoader)[datasets[1]]
            torch.manual_seed(config['seed'])
            random.seed(config['seed'])
            learning_rate = config['learning_rate']
            # learning_rate = lr
            num_hidden = config['num_hidden']
            num_proj_hidden = config['num_proj_hidden']
            activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
            base_model = ({'GCNConv': GCNConv})[config['base_model']]
            num_layers = config['num_layers']

            drop_edge_rate_1 = config['drop_edge_rate_1']
            drop_edge_rate_2 = config['drop_edge_rate_2']
            drop_feature_rate_1 = config['drop_feature_rate_1']
            drop_feature_rate_2 = config['drop_feature_rate_2']
            # tau = config['tau']
            weight_decay = config['weight_decay']
            lower = config['lower']
            upper = config['upper']
            node_lower = config['node_lower']
            node_upper = config['node_upper']
            str1 = "_weight" if is_edge_weight else ""
            str2 = "_adaptive" if is_adaptive else ""
            str3 = "_node" if is_node_weight else ""
            str4 = "_motif" if is_motif else ""
            res_path = "./results/" + datasets[0] + "_marcia_shuffle_tau_" + str(tau) + ".txt"
            with open(res_path, 'a+') as res_file:
                res_file.write(str(config) + "\n")
            best = 1e9

            for _ in range(20):
                allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData(datasets[0])
                allylabel = []
                for item in ally:
                    allylabel.append(np.argmax(item))
                edges_index = torch.tensor(edges.astype(np.int64)).T
                edges_weight = torch.tensor(edges_weight).T
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                device='cuda:1'
                allx = torch.tensor(allx.A, dtype=torch.float32).to(device)
                ally = torch.tensor(allylabel).to(device)
                edges_index = edges_index.to(device)
                edges_weight = edges_weight.to(device)
                motifs_all = motifs_all.to(device)
                motifs_num = motifs_num.to(device)

                if is_motif:
                    # n_vectors = torch.zeros(3, 10)
                    # n_vectors = nn.init.uniform_(tensor=n_vectors, a = -5, b=5)
                    # print(n_vectors)
                    normalized_motifs_num = F.normalize(motifs_num, p=2, dim=1)
                    # print(n_vectors)
                    # normalized_motifs_num = (motifs_num - torch.min(motifs_num)) / (torch.max(motifs_num) - torch.min(motifs_num))
                    allx = torch.cat([allx, normalized_motifs_num.T], dim=1)

                # 计算节点特征mask概率
                node_weight = motifs_num.sum(dim=0)
                a = node_lower
                b = node_upper
                k = (b - a) / (torch.max(node_weight) - torch.min(node_weight))
                node_probability = a + k * (node_weight - torch.min(node_weight))
                # if is_motif:
                #     encoder = Encoder(allx.shape[1]+motifs_num.shape[0], num_hidden, activation,
                #                     base_model=base_model, k=num_layers, device=device).to(device)
                # else:
                encoder = Encoder(allx.shape[1], num_hidden, activation,
                                base_model=base_model, k=num_layers, device=device).to(device)
                model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                loss = 0
                #best = 1e9
                best_t = 0
                with open(res_path, 'a+') as res_file:
                    for epoch in range(1, 2000):
                        loss = train(model, allx, motifs_all, motifs_num, edges_index, edges_weight, is_node_weight, is_motif,
                                   is_edge_weight, is_adaptive, node_probability, lower, upper)
                        if epoch % 10 == 0:
                            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss))
                        if loss < best:
                            best = loss
                            best_t = epoch
                            cnt_wait = 0
                            #只保存模型参数
                            torch.save(model.state_dict(), './model_' + str(tau) + '_' + datasets[0] + str1 + str2 + str3 + str4 + '.pkl')
                            #保存所有模型
                            torch.save(model, './model_' + str(tau) + '_' + datasets[0]  + '_' + str(_) +'.pkl')
                            #print("更新了一次")

                        else:
                            cnt_wait += 1
                        if cnt_wait == config['patience']:
                            print('Early stopping!')
                            res_file.write(f'Epoch={epoch}, Early stopping!\n')
                            break
                    model=(torch.load('./model_' + str(tau) + '_' + datasets[0]  + '_' + str(_) +'.pkl'))
                    res,acc,nmi = train_test(model, allx, motifs_all, motifs_num, edges_index, edges_weight, ally, is_edge_weight, is_motif, final=True)
                    res_file.write(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}\n acc={acc} nmi={nmi}')
                    #res_file.write(f'ephch={_}     acc={acc[0]}    nmi={nmi}\n')
                    print(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}acc={acc[0]} nmi={nmi}')
