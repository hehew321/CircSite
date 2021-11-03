"""
Created on Wed Jun 19 17:06:48 2019
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import os
from torch.autograd import Variable
import numpy as np
import logomaker as lm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import full_one_data
from dataloader import TestFullOneRna, TestFullRna
from data import to_one_hot, full_RNA_data
import tensorflow as tf
import pandas as pd
gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.train()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.model.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        gradients_as_arr = torch.autograd.grad(outputs=model_output, inputs=input_image)
        # print('gradients_as_arr: ', gradients_as_arr)
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        # print('input_image.size(): ', input_image.size())
        for i, xbar_image in enumerate(xbar_list):
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            # integrated_grads = integrated_grads + np.multiply(input_image.detach().cpu().numpy(), single_integrated_grad[0].detach().cpu().numpy()/steps)
            integrated_grads = integrated_grads + single_integrated_grad[0].detach().cpu().numpy()/steps

        return integrated_grads[0]


def _visualizeScores(array, filename, ips_positions):
    arr = np.nan_to_num(array, copy=False)
    # arr = np.maximum(arr, 0)
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
    # color_scheme = 'skylign_protein'
    logo = lm.Logo(df, figsize=(1+0.18*200, 2.08), color_scheme=color_scheme)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    logo.ax.set_ylim(min_scr * 1.08, max_scr * 1.08)
    # logo.ax.set_ylabel('contribution')

    ### code for highlighting ###
    cutoff = 0.65  # arbitrary value ; all scores above 'cutoff' % of the maximum score will be highlighted in red
    positions_to_highlight_red = [p for p in range(len(array)) if max(array[p]) > cutoff*max_scr]
    positions_to_highlight_yellow = ips_positions
    for p in positions_to_highlight_yellow:
        plt.axvspan(p - 0.5, p + 0.5, color='yellow', alpha=0.6, lw=0)
    for p in positions_to_highlight_red:
        plt.axvspan(p - 0.5, p + 0.5, color='red', alpha=0.2, lw=0)
    ### end code for highlighting ###

    plt.tight_layout()
    plt.xticks([])
    plt.savefig(filename)
    # plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    letters = "ACGU"
    window_size, steps, batch_size = 65, 100, 128
    results = []

    for rbp_name in ['HUR']:
    # for rbp_name in sorted(os.listdir('/data/wuhehe/wuhe/Bsite_data')):
        if not os.path.exists("./full_rna_integratedGradients/{}".format(rbp_name)):
            os.system("mkdir ./full_rna_integratedGradients/{}".format(rbp_name))
        data, y, counts, full_rna_names = full_RNA_data(window_size, rbp_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get params
        pretrained_model = torch.load("./models/models/{0}/{0}_lastUpdate.pkl".format(rbp_name)).to(device)
        pretrained_model.to(device)
        # test_full_data = TestFullRna(seq=data)
        # test_loader = torch.utils.data.DataLoader(dataset=test_full_data, batch_size=batch_size, shuffle=False)

        pretrained_model.eval()
        ouputs = []
        # Vanilla backprop
        IG = IntegratedGradients(pretrained_model)
        print('wwww', len(y), len(data))
        for i in range(len(data)):
            input_data = Variable(torch.from_numpy(data[i].reshape(1, 65, 4)).float()).to(device)
            # data = Variable(data)
            # output = torch.from_numpy(y[i])
            integrated_grads = IG.generate_integrated_gradients(Variable(input_data, requires_grad=True).to(device),
                                                                0, steps)
            results.append(integrated_grads[32].tolist())

        for i in range(len(full_rna_names)):
            filename = './full_rna_integratedGradients/{0}/{0}_{1}_motifs.pdf'.format(rbp_name, full_rna_names[i])
            # seq = integrated_grads
            seq = results[counts[i]+1: counts[i+1]+1]
            ips_positions = [i for i in range(len(seq))]
            _visualizeScores(seq, filename, ips_positions)
            print('{}: Integrated gradients completed.'.format(rbp_name))
            _save = pd.DataFrame({'seq': seq})
            _save.to_csv('./full_rna_integratedGradients/{0}/{0}_{1}_integratedGradients.csv'.
                         format(rbp_name, full_rna_names[i]))






