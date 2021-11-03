import torch
import random
import numpy as np
from data import get_train_data, get_test_data, to_one_hot, full_one_data, full_RNA_data
from torch.utils.data.sampler import Sampler


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, window_size, rbp_name):
        self.window_size = window_size
        self.rbp_name = rbp_name
        self.data, self.label = self.data_processing()
        super(TestDataset, self).__init__()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

    def data_processing(self):
        data, label = get_test_data(window_size=self.window_size, rbp_name=self.rbp_name)

        print('RBP: {}, Test set size {}'.format(self.rbp_name, len(label)))

        return data, label


class TrainBatchSampler(object):
    def __init__(self, pos, neg, rate, times, batch_size, window_size, shuffle=True):
        self.pos = pos
        self.neg = neg
        self.rate = rate
        self.times = times
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_size = window_size

    def __iter__(self):
        for idx in range(int(self.times*len(self.pos))):
            pos_index = np.random.choice(len(self.pos), int(self.rate * self.batch_size), replace=False)
            neg_index = np.random.choice(len(self.neg), int((1 - self.rate) * self.batch_size), replace=False)
            pos = self.pos[pos_index]
            neg = self.neg[neg_index]
            batch_data = np.concatenate((to_one_hot(pos, self.window_size), to_one_hot(neg, self.window_size)), axis=0)
            batch_label = np.array([[1.] for _ in range(len(pos_index))] + [[0.] for _ in range(len(neg_index))])
            if self.shuffle:
                index = np.random.choice(len(batch_label), len(batch_label), replace=False)
                batch_data = batch_data[index]
                batch_label = batch_label[index]

            yield torch.from_numpy(batch_data).float(), torch.from_numpy(batch_label).float()

    def __len__(self):
        return int(self.times*len(self.pos))


class TestFullOneRna(torch.utils.data.Dataset):
    def __init__(self, window_size, seq, rbp_name=''):
        self.window_size = window_size
        self.rbp_name = rbp_name
        self.seq = seq
        self.data = self.data_processing()
        super(TestFullOneRna, self).__init__()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.seq)

    def data_processing(self):
        data = full_one_data(window_size=self.window_size, seq=self.seq)
        print('Test set shape {}'.format(data.shape))
        return torch.from_numpy(data).float()


class TestFullRna(torch.utils.data.Dataset):
    def __init__(self, seq):
        self.seq = seq
        super(TestFullRna, self).__init__()

    def __getitem__(self, index):
        return self.seq[index]

    def __len__(self):
        return len(self.seq)


class RewriteBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices."""
    def __init__(self, sampler, batch_size, drop_last):
        # ...省略类型检查
        # 定义使用何种采样器Sampler
        self.sampler = sampler
        self.batch_size = batch_size
        # 是否在采样个数小于batch_size时剔除本次采样
        self.drop_last = drop_last

    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append(idx)
            # 如果采样个数和batch_size相等则本次采样完成
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        # for结束后在不需要剔除不足batch_size的采样个数时返回当前batch
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # 在不进行剔除时，数据的长度就是采样器索引的长度
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    # pos, neg = get_data(window_size=21, rbp_name='AGO1')
    # xx = DataBatchSampler(pos, neg, rate=0.2, times=10, batch_size=36, shuffle=True)
    # for i, ii in xx:
    #     print(i, ii)
    pass
