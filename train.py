import os
import torch
import random
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformer import vit_model
from torch.optim import Adam, SGD
from data import get_train_data, to_one_hot
from torch.autograd import Variable
from dataloader import TestDataset, TrainBatchSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
from occupied_gpu_memory import occupied_gpu
sys.path.append(os.path.dirname(__file__))


def train(max_stopping_step, pos_len):
    stopping_step = 0
    f_p = open('./losss/{}_loss.txt'.format(rbp_name), 'w')
    val_acc, num_train = 0.0, 0
    val_roc_auc, best_val_pr_auc = 0, 0

    for i, (data, label) in enumerate(train_data):
        model.train()
        num_train += data.shape[0]
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        optimizer.zero_grad()
        outputs = model(data)

        # loss = loss_fn(outputs, label)
        loss = BCE_L2Loss(outputs, label)
        loss.backward()
        optimizer.step()

        f_p.writelines(['i: {}, Loss: {}\n'.format(i, loss)])
        print("i: {}, Loss: {}".format(i, loss))
        if (i+1) % int(pos_len * 0.005) == 0:
            val_roc_auc, val_pr_auc, random_pr_auc = val()

            f_p.writelines(["RBP name: ", str(rbp_name), "val current PR AUC: {}, random PR AUC: {}, "
                                                         "val current ROC AUC: {}, best_val_auc: {}\n".format(
                val_pr_auc, random_pr_auc, val_roc_auc, best_val_pr_auc)])
            if val_pr_auc > best_val_pr_auc:
                stopping_step = 0
                best_val_pr_auc = val_pr_auc
                save_model(rbp_name)
            else:
                stopping_step += 1
                if stopping_step >= max_stopping_step:
                    print("Early stoping, RBP name: {}, val Accuracy: {} , val current PR AUC: {}, random PR AUC: {}, "
                          "val current ROC AUC: {}, best_val_pr_auc{}".format(rbp_name, val_acc, val_pr_auc, random_pr_auc,
                                                                           val_roc_auc, best_val_pr_auc))
                    break

    torch.save(model, './models/{0}/{0}_lastUpdate.pkl'.format(rbp_name))
    f_p.close()


def val():
    val_auc, val_loss = 0.0, 0.0
    num_val = 0
    val_acc = 0
    outputs, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_data):
            num_val = i
            data = Variable(data).to(device)
            label = Variable(label).to(device)
            output = model(data)
            loss = loss_fn(output, label)
            outputs.extend(output.data.cpu())
            labels.extend(label.data.cpu())
            val_loss += loss.data.cpu()
    val_roc_auc = roc_auc_score(labels, outputs)
    precision, recall, thresholds = precision_recall_curve(labels, outputs)

    val_pr_auc = auc(recall, precision)
    random_pr_auc = random_prauc(labels)
    print("Val: val_accuracy: {} , val_roc_auc: {}, val_pr_auc: {}, random_pr_auc: {},  val_loss: {}".format(
        val_acc, val_roc_auc, val_pr_auc, random_pr_auc, val_loss/num_val))
    return val_roc_auc, val_pr_auc, random_pr_auc


def inference(RBP):
    preds_score, labels = [], []
    # model = torch.load('./models/{0}/{0}_best_weight.pkl'.format(RBP))
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = Variable(torch.from_numpy(to_one_hot(data, window_size=args.window_size)).float()).to(device)
            label = Variable(label.float()).to(device)
            preds = model(data)
            # prediction = torch.argmax(preds.data, 1)
            preds_score.extend(preds.data.cpu().numpy().reshape(-1,))
            # preds_acc.extend(prediction)
            labels.extend(label.data.cpu().numpy().reshape(-1,))
    roc_auc = roc_auc_score(labels, preds_score)
    precision, recall, thresholds = precision_recall_curve(labels, preds_score)
    pr_auc = auc(recall, precision)
    random_pr_auc = random_prauc(labels)

    processed_preds, processed_label, left, right = filtering_smoothing(preds_score, labels, plot=False)

    return roc_auc, pr_auc, random_pr_auc, processed_preds, processed_label, preds_score, labels


def save_model(RBP):
    if not os.path.exists('./models/{}'.format(rbp_name)):
        os.mkdir('./models/{}'.format(rbp_name))
    torch.save(model, "./models/{0}/{0}_best_weight.pkl".format(RBP))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def split_train_val(data, rate):
    if isinstance(len(data)*rate, int) and isinstance(len(data)*(1-rate), int):
        len_train = len(data)*rate
        len_val = len(data)*(1-rate)
    else:
        len_train = int(len(data)*rate) + 1
        len_val = int(len(data)*(1-rate))
    return len_train, len_val


def random_prauc(labels):
    """
    Random situation
    """
    random_preds = [random.random() for _ in range(len(labels))]
    random_precision, random_recall, _ = precision_recall_curve(labels, random_preds)
    random_pr_auc = auc(random_recall, random_precision)
    return random_pr_auc


def filtering_smoothing(inputs, y, plot=False):

    _font1 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 14}
    if plot:
        # plt.subplot(1, 3, 1)
        plt.plot([i for i in range(len(inputs))], inputs)
        plt.scatter([i for i in range(len(inputs))], y, c='r', marker='*')
        plt.tick_params(labelsize=14)
        # plt.title(rna_name)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.show()

    """
    中值滤波
    """
    windows = 46
    output = []
    for i in range(len(inputs)):
        if i < windows:
            tem = sorted(inputs[:i + windows // 2])
        elif i > len(inputs) - windows:
            tem = sorted(inputs[i - windows // 2:])
        else:
            tem = sorted(inputs[i - windows // 2: i + windows // 2])

        output.append(tem[len(tem) // 2])
    if plot:
        # plt.subplot(1, 3, 2)
        plt.plot([i for i in range(len(output))], output)
        plt.scatter([i for i in range(len(output))], y, c='r', marker='*')
        plt.tick_params(labelsize=14)
        # plt.title(rna_name)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.show()
    """
    动态阈值
    """
    output = np.array(output)
    scale = 23
    mid = scale // 2
    # threshold = 0.5

    for j in range(len(output) - mid):
        for threshold in np.linspace(0.5, 1, 11):
            if j < mid and all(output[:j + mid] > threshold):
                output[:j + mid] = 1.
            elif j > len(output) - mid and all(output[j - mid:] > threshold):
                output[j - mid:] = 1.
            elif mid <= j <= len(output) - mid and all(output[j - mid:j + mid] > threshold):
                output[j - mid:j + mid] = 1.
    left, right = [], []
    for i in range(len(output)):
        if output[i] < 1.:
            output[i] = 0.
    for i in range(1, len(output)-1):
        if output[i] == 1.:
            if output[i-1] == 0.:
                left.append(i)
            elif output[i+1] == 0.:
                right.append(i)
    if plot:
        # plt.subplot(1, 3, 3)
        plt.plot([i for i in range(len(output))], output)
        plt.scatter([i for i in range(len(output))], y, c='r', marker='*')
        # plt.title(rna_name)
        plt.tick_params(labelsize=14)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.show()
    return output, y, left, right


def L2Loss(model, alpha=1e-5):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.2*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss


def BCE_L2Loss(y_pred, y_true):
    bce_loss = torch.nn.BCELoss()(y_pred, y_true)
    l2_loss = L2Loss(model)
    total_loss = bce_loss + l2_loss
    return total_loss


def parse_arguments(parser):
    parser.add_argument('--seed', type=int, default=1024, help='xx')
    parser.add_argument('--embed_dim', type=int, default=384, help='xx')
    # parser.add_argument('--rbp_name', type=str, default='TAF15', help='xx')
    parser.add_argument('--window_size', type=int, default=65, help='xx')
    parser.add_argument('--b_size', type=int, default=128, help='xx')
    # parser.add_argument('--n_epochs', type=int, default=50, help='xx')
    parser.add_argument('--max_stopping_step', type=int, default=5, help='xx')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    occupied_gpu(cuda_device='0', rata=0.15)
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    seed_everything(args.seed)
    # name, aucs = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # , 'TIAL1', 'TIA1', 'AGO1', 'AGO2', 'AGO3'
    for rbp_name in ['ALKBH5']:
    # for rbp_name in sorted(os.listdir('/data/wuhehe/wuhe/Bsite_data')):
        print('RBP name: ', rbp_name)
        train_pos, train_neg = get_train_data(window_size=args.window_size, rbp_name=rbp_name)
        pos_len = train_pos.shape[0]
        print('train_pos.shape: {}, train_neg.shape: {}'.format(train_pos.shape, train_neg.shape))
        train_data = TrainBatchSampler(train_pos[: int(0.9*len(train_pos))], train_neg[: int(0.9*len(train_neg))],
                                       rate=0.2, times=5, batch_size=args.b_size, shuffle=True,
                                       window_size=args.window_size)

        val_data = TrainBatchSampler(train_pos[int(0.9*len(train_pos)):], train_neg[int(0.9*len(train_neg)):],
                                     rate=0.2, times=0.005, batch_size=args.b_size, shuffle=True,
                                     window_size=args.window_size)
        del train_neg
        del train_pos

        model = vit_model(d_model=args.embed_dim, dropout=0., max_len=args.window_size+1)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        loss_fn = torch.nn.BCELoss()

        train(max_stopping_step=args.max_stopping_step, pos_len=pos_len)
        del train_data
        del val_data
        print('{} start test {}'.format('*' * 10, '*' * 10))
        test_data = TestDataset(args.window_size, rbp_name)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.b_size, shuffle=False)
        roc_auc, pr_auc, random_pr_auc, processed_preds, processed_label, preds_score, labels = inference(rbp_name)
        print('RBP: {}, test set roc_auc: {}, pr_auc: {}, random_pr_auc: {}'
              .format(rbp_name, roc_auc, pr_auc, random_pr_auc))
        del test_data
        del test_loader
        del model
        acc = accuracy_score(processed_label, processed_preds)
        precision, recall = precision_score(processed_label, processed_preds), recall_score(processed_label, processed_preds)
        f1 = f1_score(processed_label, processed_preds)
        f = open('./models/{}_auc.txt'.format(rbp_name), 'w')
        f.writelines([str(rbp_name), ' roc_auc: ', str(roc_auc), ' pr_auc: ', str(pr_auc),
                      ' random_pr_auc: ', str(random_pr_auc), ' acc: ', str(acc),
                      ' precision: ', str(precision), ' recall: ', str(recall),
                      ' f1: ', str(f1)])
        f.close()
        _save = pd.DataFrame({'preds_score': preds_score, 'processed_preds': processed_preds, 'labels': labels})
        _save.to_csv('./models/{}_preds.csv'.format(rbp_name), index=False)

