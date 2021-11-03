from occupied_gpu_memory import occupied_gpu
occupied_gpu(cuda_device='1', rata=0.15)
import torch
import os, sys
import logomaker as lm

# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import pandas as pd
from dataloader import TestFullOneRna, TestFullRna
from data import to_one_hot, full_RNA_data, t_TSNE
from torch.autograd import Variable
sys.path.append(os.path.dirname(__file__))

import torch
import time, os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_embedding(protein_name, data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i]),
        #          fontdict={'weight': 'bold', 'size': 9})
    y = []
    for i in range(len(label)):
        if label[i] == 1.:
            y.append(0.)
        else:
            y.append(1.)
    plt.scatter(data[:, 0], data[:, 1], color=plt.cm.Set1(label), marker='*')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    path = './tsne/{0}/{0}.pdf'.format(protein_name)
    plt.savefig(path)
    plt.show()
    print('xxxx complete')


def plot_tsne(embedding, label, protein_name):
    '''t-SNE'''
    t0 = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(embedding)  # 转换后的输出
    t1 = time.time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    print(result)
    plot_embedding(protein_name, result, label,
                   '{}'.format(protein_name))


def get_embedding(net, input_data):
    with torch.no_grad():
        x = input_data.permute(0, 2, 1)
        x = net.rna_embed(x)
        x = x.permute(0, 2, 1)
        x1, _ = net.gru(x)
        x1 = x1[:, -1, :]
        x1 = net.linear1(x1)

        cls_token = net.cls_token.expand(x.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x), dim=1)
        x2 = net.pos_drop(x2 + net.pos_embed(x2))
        x2 = net.blocks(x2)
        x2 = net.norm(x2)
        x2 = net.pre_logits(x2[:, 0])
        x2 = net.head(x2)
        x = torch.cat([x1, x2], dim=1)
    return x.cpu().numpy()


def get_iC_embedding(net, input_data):
    with torch.no_grad():
        x = input_data.permute(0, 2, 1)
        x = net.rna_embed(x)
        x = x.permute(0, 2, 1)
        x1, _ = net.gru(x)
        x1 = x1[:, -1, :]
        x1 = net.linear1(x1)

        cls_token = net.cls_token.expand(x.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x), dim=1)
        x2 = net.pos_drop(x2 + net.pos_embed(x2))
        x2 = net.blocks(x2)
        x2 = net.norm(x2)
        x2 = net.pre_logits(x2[:, 0])
        x2 = net.head(x2)
        x = torch.cat([x1, x2], dim=1)
    return x.cpu().numpy()


if __name__ == '__main__':
    window_size = 65
    remove_rbp = []
    for rbp_name in ['EIF4A3']:
    # for rbp_name in sorted(os.listdir('/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site')):
        if rbp_name in remove_rbp:
            continue
        if not os.path.exists("./tsne/{}".format(rbp_name)):
            os.system("mkdir ./tsne/{}".format(rbp_name))
        print('rbp_name', rbp_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = torch.load(
        #     "/data/wuhehe/wuhe/cnnLstm_transformer/models/models/format_lastUpdata2_rbp-pkl/{0}.pkl".format(
        #         rbp_name), map_location=lambda storage, loc: storage)
        model = torch.load(
            "/data/wuhehe/wuhe/cnnLstm_transformer/models/models/format_lastUpdata2_rbp-pkl/{0}.pkl".format(
                rbp_name, map_location={'cuda:0': 'cuda:1'})).to(device)
        # print(model)
        print('*'*100)

        # model = torch.load('/data/wuhehe/wuhe/cnnLstm_transformer/models_save/last_update/QKI.pkl')
        data, label, counts, full_rna_names = t_TSNE(window_size, rbp_name)

        model.eval()

        embeds = get_embedding(model, input_data=Variable(torch.from_numpy(data).float()).to(device))
        # print(embeds)
        try:
            plot_tsne(embedding=embeds, label=label, protein_name=rbp_name)
        except:
            pass
        torch.cuda.empty_cache()



