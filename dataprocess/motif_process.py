import pickle as pkl
import torch
import sys
import numpy as np
from collections import Counter
def loadAllData(dataset_str):
    names = ['allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./datasets/" + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    return tuple(objects)

dataset_str = "amazon-photo"

if __name__ == "__main__":
    # loadAllData(dataset_str)
    with open("../data/" + "ind.{}.motif.dict".format(dataset_str), 'rb') as f:
        if sys.version_info > (3, 0):
            motif_dict = pkl.load(f, encoding='latin1')
        else:
            motif_dict = pkl.load(f)
    # motifs_all = torch.zeros(len(motif_dict[0]), len(motif_dict), len(motif_dict))
    motifs_num = torch.zeros(len(motif_dict[0]), len(motif_dict))
    for key in motif_dict:
        motifs_type = motif_dict[key]
        # temp = []
        for i in range(len(motifs_type)):
            motifs = motifs_type[i]
            # temp1 = []
            if len(motifs) > 0:
                motifs_num[i][key] = len(motifs)
            else:
                motifs_num[i][key] = 1
            # temp1 += list(np.array(tuple(motifs)).flatten())
            # temp.append(temp1)
        # for t in range(len(temp)):
        #     node_num = Counter(temp[t])
        #     for node in node_num:
        #         motifs_all[t][key][node] += node_num[node]
    # with open("../data/" + "ind.{}.motifs.all".format(dataset_str), "wb") as f:
    #     pkl.dump(motifs_all, f)
    print("aaaaaaa")
    with open("../data/" + "ind.{}.motifs.num".format(dataset_str), "wb") as f:
        pkl.dump(motifs_num, f)