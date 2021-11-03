import torch
import os, sys
import logomaker as lm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.autograd import Variable
from modelX import vit_model, VisionTransformer, PositionalEncoding
sys.path.append(os.path.dirname(__file__))


class TestFullRna(torch.utils.data.Dataset):
    def __init__(self, seq):
        self.seq = seq
        super(TestFullRna, self).__init__()

    def __getitem__(self, index):
        return self.seq[index]

    def __len__(self):
        return len(self.seq)


def full_RNA_data(window_size, seq):
    """
    The format of seq is ['sequence 1 ','sequence 2']
    """
    len_seq, _data, counts = [-1], [], [-1]
    count = 0
    for i in range(len(seq)):
        _seq = seq[i]
        len_seq.append(len(_seq))
        for j in range(len(_seq)):
            if j < window_size // 2:
                _data.append(
                    (window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
            elif j >= len(_seq) - window_size // 2:
                _data.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
            else:
                _data.append(_seq[j - window_size // 2: j + window_size // 2 + 1])
            count += 1

        counts.append(count)
    return to_one_hot(_data, window_size=window_size), np.array(counts)


def to_one_hot(seq, window_size):
    seq_data = []
    for i in range(len(seq)):
        mat = np.array([0.] * 4 * window_size).reshape(window_size, 4)
        for j in range(len(seq[i])):
            # print(seq[i])
            if seq[i][j] == 'A':
                mat[j][0] = 1.0
            elif seq[i][j] == 'C':
                mat[j][1] = 1.0
            elif seq[i][j] == 'G':
                mat[j][2] = 1.0
            elif seq[i][j] == 'U' or seq[i][j] == 'T':
                mat[j][3] = 1.0
            elif seq[i][j] == 'N':
                mat[j] = 0.25
            elif seq[i][j] == '0':
                mat[j] = 0
            else:
                print('seq[i][j]', seq[i][j])
                print("Presence of unknown nucleotides")
                sys.exit()
        seq_data.append(mat)
    return np.array(seq_data)


def filtering_smoothing(preds, input_seq_name, thres=0.5, plot=False):

    _font1 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 14}
    print(preds)
    if plot:
        plt.plot([i for i in range(len(preds))], preds)
        plt.tick_params(labelsize=14)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.title('./{0}_before-filtering.pdf'.format(input_seq_name))
        plt.savefig('./{0}_before-filtering.pdf'.format(input_seq_name))
        # plt.show()
        plt.close()

    """
    中值滤波
    """
    windows = 46
    output = []
    for i in range(len(preds)):
        if i < windows:
            tem = sorted(preds[:i + windows // 2])
        elif i > len(preds) - windows:
            tem = sorted(preds[i - windows // 2:])
        else:
            tem = sorted(preds[i - windows // 2: i + windows // 2])

        output.append(tem[len(tem) // 2])
    if plot:
        plt.plot([i for i in range(len(output))], output)
        plt.tick_params(labelsize=14)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.title('./{0}_after-filtering.pdf'.format(input_seq_name))
        plt.savefig('./{0}_after-filtering.pdf'.format(input_seq_name))
        # plt.show()
        plt.close()
    """
    二值化决策
    """
    preds = np.array(output)
    scale = 23
    mid = scale // 2
    for j in range(mid, len(preds) - mid):
        if all(preds[j - mid:j + mid] > thres):
            preds[j - mid:j + mid] = 1.

    preds[preds != 1.] = 0
    if plot:
        plt.plot([i for i in range(len(preds))], preds)
        plt.tick_params(labelsize=14)
        plt.ylabel('Score', _font1)
        plt.xlabel('Location on full-RNA', _font1)
        plt.title('./{0}_after-binary_decision.pdf'.format(input_seq_name))

        plt.savefig('./{0}_after-binary_decision.pdf'.format(input_seq_name))
        # plt.show()
        plt.close()

    return output


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.train()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.model.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps+1)/steps
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        gradients_as_arr = torch.autograd.grad(outputs=model_output, inputs=input_image)
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = np.zeros(input_image.size())
        for i, xbar_image in enumerate(xbar_list):
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad[0].detach().cpu().numpy()/steps

        return integrated_grads[0]


def _visualizeScores(array, filename, ips_positions):
    arr = np.nan_to_num(array, copy=False)
    vals = arr
    df = pd.DataFrame(vals)
    df.columns = list(letters)
    max_scr = np.max(vals)
    min_scr = np.min(vals)
    color_scheme = {"G": "#FBB116",
                    "A": "#0C8040",
                    "C": "#34459C",
                    "U": "#CB2026"
                    }
    logo = lm.Logo(df, figsize=(1+0.18*200, 2.08), color_scheme=color_scheme)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    logo.ax.set_ylim(min_scr * 1.08, max_scr * 1.08)

    cutoff = 0.65
    positions_to_highlight_red = [p for p in range(len(array)) if max(array[p]) > cutoff*max_scr]
    positions_to_highlight_yellow = ips_positions
    for p in positions_to_highlight_yellow:
        plt.axvspan(p - 0.5, p + 0.5, color='yellow', alpha=0.6, lw=0)
    for p in positions_to_highlight_red:
        plt.axvspan(p - 0.5, p + 0.5, color='red', alpha=0.2, lw=0)
    plt.tight_layout()
    plt.xticks([])
    plt.show()
    plt.savefig(filename)



if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    window_size = 65
    name_threshold = {'AGO1': 0.13, 'AGO2': 0.1, 'AGO3': 0.16, 'ALKBH5': 0.22, 'AUF1': 0.57, 'C17ORF85': 0.86,
                      'C22ORF28': 0.71, 'CAPRIN1': 0.51, 'DGCR8': 0.12, 'EIF4A3': 0.1, 'EWSR1': 0.43,
                      'FMRP': 0.16, 'FOX2': 0.86, 'FUS': 0.09, 'FXR1': 0.72, 'FXR2': 0.50,
                      'HNRNPC': 0.23, 'HUR': 0.11, 'IGF2BP1': 0.12, 'IGF2BP2': 0.13, 'IGF2BP3': 0.12,
                      'LIN28A': 0.51, 'LIN28B': 0.29, 'METTL3': 0.36, 'MOV10': 0.10, 'PTB': 0.10,
                      'PUM2': 0.29, 'QKI': 0.58, 'SFRS1': 0.11, 'TAF15': 0.78, 'TDP43': 0.44,
                      'TIA1': 0.30, 'TIAL1': 0.11, 'TNRC6': 0.15, 'U2AF65': 0.65, 'WTAP': 0.78,
                      'ZC3H7B': 0.29}
    """
    用户输入
    rbp_name， seq， plot
    """
    rbp_name = 'AGO1'
    seq = ['AGACAGTCCATGGCAAAACTTTCAGTGTTGGGTTCTGCCTCCTGCTCAGTTCAGAAAGAGATGGAATACAGACTATCTAATTCCTTTCTCGTCTAAACTTAACATTGCTGCGAAAGTTAATTTTTTAGCCTATTCAGAAGTGCTGACTGATAACTTAAAAGTTGGAAGCTTTTATAAAACATATTCAAGGATACTTTTTGATTTAATGGAACTGGCTATTTGAGAAGTGTTTGAAACTTTTGCCATGGCTGCAGGACTTACATTCTTTTTTGGGAGGACGGTGGGGAGACAGGGAGTGGTAAAGGGAAAAGGTTAAAAATCCACCTGTGGTTGTATATTCTTCTATTCTGTCACTCTGTTACCTAGACTGTGAGAGGCTTTTGCCTTCAGTCAGATTAAAAAGAGCAGGGCCTAACATTGAGTGATAGCACCTGCTTTGATAAATAGGTTTTCTCACTCTTCTTTTTTTCCTTCTTTTATCCCTCACTCCCTCCCCTAAACCCTGCTTCAGCACAATGGACTAATTCTAGCATTCTGATCATAAGGCCCTCCATTTTCCTAATGTGTTTCAAGGAATCTTTTTAGGAAAAATATCCAGATTATTCATCCACTTTTTTTAGTATCTACTAACAACTCCTTTTTTTCTCTAGAGAGTTATGAAGGAACAGGTTGTCCTTGTCTGGAGTCAAGCTAAACACATGATTTGTTTTATCAGCAGCTGGAGCAGAAGTTGAAAATGTCTTTCTGTGAGACAGTAATTTGCTACTGAAGCTTTATGGCTTGTTTGCACTGATTACTCCAGGATCCAAAAACTTGGTGAAAGTCACTGAAACACTCAAGGCAAATTACTTTACAGCCCTGAGTGTCTGTCACCATAGTTTGCATAATGAATATGAATCCCATTGGTGTGTGATGTAGGAAATCCTGTAGTTGTATTTTCTTGAACTGAAATATTTGACTCAAAATAATTAAGACTCATTGTCATTTTTCATCTTGGCATTATTGTGGACAAGTTGACATATTAAATCTCTTTGCTTTCTGGTAAGCTTAGCTTTTAAAATGCATTTTCCCTTGTCCTGTCTTTAACTAGATATACATGCTTATATTTATAGTGGGTTTCACAGACTATAAAATTGAATGTATGAAATTTTTATTTATATCAGTGCTTTTAATAATGAAGATATTTTTGGAGTAATGGTGCTGTCTTGTAGCGAGTTATTAATCATAGTAAGATTTTTTTCTCTTCATTTGCTTTTTTT']
    plot = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = torch.load("/data/wuhehe//wuhe/cnnLstm_transformer/models/models/format_lastUpdata2_rbp-pkl/{0}.pkl".format(rbp_name)).to(device)
    model = torch.load('/data/wuhehe/wuhe/cnnLstm_transformer/models_save/last_update/QKI.pkl')
    data, counts = full_RNA_data(window_size, seq)
    test_full_data = TestFullRna(seq=data)
    test_loader = torch.utils.data.DataLoader(dataset=test_full_data, batch_size=2048, shuffle=False)

    model.eval()
    ouputs = []
    with torch.no_grad():
        for _data in test_loader:
            _data = Variable(_data.float()).to(device)
            ouput = model(_data)
            ouputs.extend(ouput.data.cpu().numpy().reshape(-1,).tolist())
    for i in range(len(seq)):
        output = filtering_smoothing(preds=ouputs[counts[i]+1:counts[i+1]+1], input_seq_name=rbp_name+'_'+str(i),
                                     plot=plot, thres=name_threshold[rbp_name])
    """
    Start drawing motifs
    """
    letters = "ACGU"
    ouputs, integrated_result = [], []
    steps = 100
    model.train()
    IG = IntegratedGradients(model)
    for i in range(len(data)):
        input_data = Variable(torch.from_numpy(data[i].reshape(1, 65, 4)).float()).to(device)
        integrated_grads = IG.generate_integrated_gradients(Variable(input_data, requires_grad=True).to(device),
                                                            0, steps)
        integrated_result.append(integrated_grads[32].tolist())
    for i in range(len(seq)):
        save_fileName = './{0}_motifs.pdf'.format(rbp_name+'_'+str(i))
        seq = integrated_result[counts[i] + 1: counts[i + 1] + 1]
        ips_positions = [i for i in range(len(seq))]
        _visualizeScores(seq, save_fileName, ips_positions)






