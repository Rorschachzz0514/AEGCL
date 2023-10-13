import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.nn as dglnn
import dgl.function as fn
import pickle as pkl
import sys
import numpy as np



import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, device = None):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.device = device
        self.conv = [base_model(in_channels, 2 * out_channels)]
        # for _ in range(1, k-1):
        # self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, edge_index, edge_weight = None, motifs_all = None, motifs_num = None):
        # for i in range(self.k):
            # temp = [(torch.mm(motifs_all[mat], x).T / motifs_num[mat]).T for mat in range(len(motifs_all))]
            # temp2 = torch.stack(temp, 0)
            # x_agg = torch.mean(temp2, 0).to(self.device)
        if edge_weight == None:
            x = self.activation(self.conv[0](x, edge_index))
            # x = torch.cat([x, motifs_num.T], dim=1)
            x = self.activation(self.conv[1](x, edge_index))
            # x = self.activation(self.conv[2](x, edge_index))
        else:
            x = self.activation(self.conv[0](x, edge_index, edge_weight))
            # x = torch.cat([x, motifs_num.T], dim=1)
            x = self.activation(self.conv[1](x, edge_index, edge_weight))
            # x = self.activation(self.conv[2](x, edge_index, edge_weight))
        return x


class MARCIA(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor, motifs_all = None, motifs_num = None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight, motifs_all, motifs_num)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x





objects=[]
dataset="polblogs"
with open("../data/ind."+dataset+".graph.shuffle", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

with open("../data/ind."+dataset+".allx.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))

with open("../data/ind."+dataset+".ally.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))
with open("../data/ind."+dataset+".edges.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))
with open("../data/ind."+dataset+".edges.weight.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))

with open("../data/ind."+dataset+".motifs.all.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))

with open("../data/ind."+dataset+".motifs.num.shuffle", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))

U,V=[],[]
for u,v in objects[0].items():
    for vv in v:
        U.append(u)
        V.append(vv)
        print(f'u:{u}')
        print(f'v:{vv}')
g = dgl.graph((U, V))
g.ndata['feat']=torch.tensor(objects[1].toarray()).type(torch.float32)
g.ndata['label']=torch.tensor([np.argmax(ally)for ally in objects[2]])
print(objects)
#
# dataset = dgl.data.CoraGraphDataset()
# graph = dataset[0]
# #graph = g
#
# print('Node features')
# print(graph.ndata)
# print('Edge features')
# print(graph.edata)
# u, v = graph.edges()
# #graph=g
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']
#
def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())
#
# class SAGE(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats):
#         super().__init__()
#         # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
#         self.conv1 = dglnn.SAGEConv(
#             in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
#         self.conv2 = dglnn.SAGEConv(
#             in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
#
#     def forward(self, graph, inputs):
#         # 输入是节点的特征
#         h = self.conv1(graph, inputs)
#         h = F.relu(h)
#         h = self.conv2(graph, h)
#         return h
# class Model(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features,marcia:MARCIA):
#         super().__init__()
#         self.marica:MARCIA = marcia
#         self.pred = DotProductPredictor()
#     def forward(self, g, neg_g, x):
#         h = self.sage(g, x)
#         return self.pred(g, h), self.pred(neg_g, h)
# def compute_loss(pos_score, neg_score):
#     # 间隔损失
#     n_edges = pos_score.shape[0]
#     return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
#
# node_features = graph.ndata['feat']
# n_features = node_features.shape[1]
k = 5
tau=0.4
#datasets=[]
#datasets[0]='acmv9'
marcia = (torch.load('../model/model_' + str(tau) + '_' + 'polblogs_19' + '.pkl'))
allx=torch.tensor(objects[1].A, dtype=torch.float32).to('cuda:0')
edges=objects[3]
edge_index = torch.tensor(edges.astype(np.int64)).T.to('cuda:0')
edge_weight=objects[4]
edge_weight = torch.tensor(edge_weight).T.to('cuda:0')
motifs_all=objects[5].to('cuda:0')
motifs_num=objects[6].to('cuda:0')
normalized_motifs_num = F.normalize(motifs_num, p=2, dim=1)
allx = torch.cat([allx, normalized_motifs_num.T], dim=1)
z = marcia(allx, edge_index, edge_weight, motifs_all, motifs_num)
#
# model = Model(marcia=macia)
# opt = torch.optim.Adam(model.parameters())
# for epoch in range(10):
#     negative_graph = construct_negative_graph(graph, k)
#     pos_score, neg_score = model(graph, negative_graph, node_features)
#     loss = compute_loss(pos_score, neg_score)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print(loss.item())
#
#
#h=model.sage(graph,graph.ndata['feat'])
h=z
from sklearn.metrics import roc_auc_score
pred=DotProductPredictor()
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)
with torch.no_grad():
    pos_score = pred(g, h.to('cpu'))
    neg_score = pred(construct_negative_graph(g, k), h.to('cpu'))
    print('AUC', compute_auc(pos_score, neg_score))
# ————————————————
# 版权声明：本文为CSDN博主「CHEONG_KG」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/feilong_csdn/article/details/117150872