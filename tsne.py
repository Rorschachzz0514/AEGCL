import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import torch
from sklearn.manifold import TSNE

def plot_embeddings(embeddings, Features, Labels):

    # norm = Normalized(embeddings)
    # embeddings = norm.MinMax()

    emb_list = []
    for k in range(Features.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Features.shape[0]):
        color_idx.setdefault(Labels[i], [])
        color_idx[Labels[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s = 5) # c=node_colors)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.show()

if __name__ == '__main__':
    method = "GRACE"
    dataset = "Polblogs"
    embedding = torch.load('D:\\论文写作\\RES\\tSNE\\'+method+'\\'+dataset+'\\embedding.pt',map_location='cpu').detach().numpy()
    # embedding = embedding.reshape(-1,256)
    labels = torch.load('D:\\论文写作\\RES\\tSNE\\'+method+'\\'+dataset+'\\label.pt',map_location='cpu').detach().numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(embedding)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(X_norm)
    color_idx = {}
    for i in range(X_norm.shape[0]):
        color_idx.setdefault(labels[i], [])
        color_idx[labels[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=5)  # c=node_colors)
    # plt.axis('off')
    # plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(top=False, bottom=False, left=False, right=False)
    plt.gca.legend_ = None
    plt.savefig(method + '.png')
    plt.show()
    print('done!')