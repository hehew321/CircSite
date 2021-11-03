import os
import torch
import time
from data import to_one_hot
import pandas as pd
from torch.autograd import Variable
from dataloader import TestDataset
import matplotlib.pyplot as plt
from tool import filtering_smoothing, random_prauc
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

start_time = time.time()
window_size, b_size = 65, 128
print('{} start test {}'.format('*' * 10, '*' * 10))
RPBs, roc_aucs, pr_aucs, random_pr_aucs, accs, precisions, recalls, f1s = [], [], [], [], [], [], [], []
# for rbp_name in sorted(os.listdir('/data/wuhehe/wuhe/Bsite_data_20000')):
for rbp_name in ["MOV10"]:
    # model = torch.load('./models/last_update/{}.pkl'.format(rbp_name), map_location={'cuda:1': 'cuda:0'})
    model = torch.load('/data/wuhehe/wuhe/cnnLstm_transformer/models_save/last_update/{}.pkl'.format(rbp_name))
    # model = torch.load('./models/AGO1/AGO1_17471_0.8470.pkl', map_location={'cuda:0': 'cuda:1'})
    # model = torch.load('./models/AGO1/RbpBSite_0.8343.pkl', map_location={'cuda:1': 'cuda:0'})
    # model = torch.load('./models/AGO1/RbpBSite_0.8343.pkl', map_location={'cuda:0': 'cuda:1'})

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_data = TestDataset(window_size, rbp_name, b_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=b_size, shuffle=False)

    preds_score, labels = [], []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = Variable(torch.from_numpy(to_one_hot(data, window_size=65)).float()).to(device)
            label = Variable(label.float()).to(device)
            preds = model(data)
            preds_score.extend(preds.data.cpu())
            labels.extend(label.data.cpu())

    roc_auc = roc_auc_score(labels, preds_score)
    precision, recall, thresholds = precision_recall_curve(labels, preds_score)
    pr_auc = auc(recall, precision)
    random_pr_auc = random_prauc(labels)
    print('RBP: {},  roc_auc: {},  pr_auc: {}, random_pr_auc: {}'.format(rbp_name, roc_auc, pr_auc, random_pr_auc))

    # processed_preds, labels, left, right = filtering_smoothing(preds_score, labels, plot=False)
    # precision, recall = precision_score(labels, processed_preds), recall_score(labels, processed_preds)
    # acc = accuracy_score(labels, processed_preds)
    # f1 = f1_score(labels, processed_preds)
    # print('RBP: {},  roc_auc: {},  pr_auc: {}, random_pr_auc: {}, acc: {}, precision: {}, '
    #       ' recall: {}, f1: {}'.format(rbp_name, roc_auc, pr_auc, random_pr_auc, acc,
    #                                    precision, recall, f1))
    # RPBs.append(rbp_name)
    # roc_aucs.append(roc_auc)
    # pr_aucs.append(pr_auc)
    # random_pr_aucs.append(random_pr_auc)
    # accs.append(acc)
    # precisions.append(precision)
    # recalls.append(recall)
    # f1s.append(f1)
# _save = pd.DataFrame({'rbp_name': RPBs, 'roc_auc': roc_aucs, 'pr_auc': pr_aucs, 'random_pr_auc': random_pr_aucs,
#                       'acc': accs, 'precision': precisions, 'recall': recalls, 'f1': f1s})
# _save.to_csv('get_indicators.csv', index=False)
