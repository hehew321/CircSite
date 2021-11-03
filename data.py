import sys, random
import numpy as np


def get_train_data(window_size, rbp_name):
    """
    get data and label
    """
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i + 1].strip()
    f_name_seq.close()
    f_p = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_train.fasta".format(rbp_name))
    print('/cnnlstm_transformer/../data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site')
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()
    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq
    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:]][int(line[-2])-1:int(line[-1])] = 1.

    rna_name = list(name_y.keys())
    assert len(name_y) == len(name_seq)
    data_pos, data_neg = [], []
    for i in range(len(rna_name)):
        _seq = name_seq[rna_name[i]]
        _y = name_y[rna_name[i]]

        for j in range(len(_seq)):
            if _y[j] == 1:
                if j < window_size // 2:
                    data_pos.append(
                        (window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
                elif j >= len(_seq) - window_size // 2:
                    data_pos.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
                else:
                    data_pos.append(_seq[j - window_size // 2: j + window_size // 2 + 1])
            else:
                if j < window_size // 2:
                    data_neg.append(
                        (window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
                elif j >= len(_seq) - window_size // 2:
                    data_neg.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
                else:
                    data_neg.append(_seq[j - window_size // 2: j + window_size // 2 + 1])
    f_p.close()
    return np.array(data_pos), np.array(data_neg)


def get_test_data(window_size, rbp_name):
    """
    get data and label
    """
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i + 1].strip()
    f_name_seq.close()

    f_p = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_test.fasta".format(rbp_name))
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()
    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq
    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:]][int(line[-2])-1:int(line[-1])] = 1.

    rna_name = list(name_y.keys())
    assert len(name_y) == len(name_seq)
    data, label = [], []
    for i in range(len(rna_name)):
        _seq = name_seq[rna_name[i]]
        _y = name_y[rna_name[i]]

        for j in range(len(_seq)):
            if _y[j] == 1:
                label.append([1])
            else:
                label.append([0])
            if j < window_size // 2:
                data.append(
                    (window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
            elif j >= len(_seq) - window_size // 2:
                data.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
            else:
                data.append(_seq[j - window_size // 2: j + window_size // 2 + 1])
    f_p.close()
    return np.array(data), np.array(label)


def full_test_data(window_size, rbp_name):
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i + 1].strip()
    f_name_seq.close()

    f_p = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_test.fasta".format(rbp_name))
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()

    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq

    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:17]][int(line[-2]) - 1:int(line[-1])] = 1.

    _data = []
    _y = []
    len_seq = []
    name = []
    rna_name = list(name_y.keys())
    for i in range(len(rna_name)):
        _seq = name_seq[rna_name[i]]
        name.append(rna_name[i])
        _y.extend(name_y[rna_name[i]])
        len_seq.append(len(_seq))

        for j in range(len(_seq)):
            if j < window_size // 2:
                _data.append(
                    (window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
            elif j >= len(_seq) - window_size // 2:
                _data.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
            else:
                _data.append(_seq[j - window_size // 2: j + window_size // 2 + 1])

    _data = to_one_hot(_data, window_size=window_size)
    return _data, _y, len_seq, name


def full_one_data(window_size, seq, label):
    # AUF1 hsa_circ_0000078
    # _seq = 'GTATTATATTTTTGTCCCGGTTTTAAATCTGGAGTAAAGCACTTATTTAATATTATTTCAAGGAAGAAAGAAGCTCTAAAGGATGAAGCCAATCAAAAAGACAAGGGAAATTGCAAAGAAGATTCTTTGGCAAGTTATGAATTGATATGCAGTTTACAGTCCTTAATCATTTCGGTTGAACAGCTCCAGGCTAGTTTTCTCTTAAATCCAGAGAAATATACTGATGAACTTGCCACTCAGCCAAGGCGACTGCTTAACACACTGAGGGAACTCAACCCTATGTATGAAGGATATCTACAGCATGATGCACAGGAAGTATTACAATGTATTTTGGGAAACATTCAAGAAACATGCCAACTCCTAAAAAAAGAAGAAGTAAAAAATGTGGCAGAATTACCTACTAAGGTAGAAGAAATACCTCATCCGAAAGAGGAAATGAATGGTATTAACAGCATAGAGATGGACAGTATGAGGCATTCTGAAGACTTTAAAGAGAAACTCCCAAAAGGAAATGGGAAAAGAAAAAGTGACACTGAATTTGGTAACATGAAGAAAAAAGTTAAATTATCCAAGGAACACCAGTCATTGGAAGAGAACCAGAGACAAACTAGATCAAAAAGAAAAGCTACAAGTGATACATTAGAGAGTCCTCCTAAAATAATTCCCAAGTATATTTCTGAAAATGAGAGTCCAAGACCCTCACAAAAGAAATCAAGAGTTAAAATAAATTGGTTAAAGTCTGCAACTAAGCAACCCAGCATTCTTTCTAAATTTTGTAGTCTGGGAAAAATAACAACAAACCAAGGAGTCAAAGGACAATCTAAAGAAAATGAATGTGATCCTGAAGAGGACTTGGGGAAGTGTGAAAGTGATAACACAACTAATGGTTGTGGACTTGAATCTCCAGGAAATACTGTTACACCTGTAAATGTTAATGAAGTTAAACCCATAAACAAAG'
    # FXR1 hsa_circ_0099182
    # _seq = 'AAAAAGATGTTAACAGAACAGATCGAACAAACAAGTTTTATGAAGGCCAAGATAATCCAGGGTTGATTTTACTTCATGACATTTTGATGACCTACTGTATGTATGATTTTGATTTAGGATATGTTCAAGGAATGAGTGATTTACTTTCCCCTCTTTTATATGTGATGGAAAATGAAGTGGATGCCTTTTGGTGCTTTGCCTCTTACATGGACCAAATGCATCAGAATTTTGAAGAACAAATGCAAGGCATGAAGACCCAGCTAATTCAGCTGAGTACCTTACTTCGATTGTTAGACAGTGGATTTTGCAGTTACTTAGAATCTCAGGACTCTGGATACCTTTATTTTTGCTTCAGGTGGCTTTTAATCAGATTCAAAAGGGAATTTAGTTTTCTAGATATTCTTCGATTATGGGAGGTAATGTGGACCGAACTACCATGTACAAATTTCCATCTTCTTCTCTGTTGTGCTATTCTGGAATCAGAAAAGCAGCAAATAATGGAAAAGCATTATGGCTTCAATGAAATACTTAAGCATATCAATGAATTGTCCATGAAAATTGATGTGGAAGATATACTCTGCAAGGCAGAAGCAATTTCTCTACAGATGGTAAAATGCAAG'
    # FXR1 hsa_circ_0099109
    # _seq = 'GTTTGCCCTTGTTGGTGTTGGATCTGAAGCATCTTCAAAAAAGTTAATGGATCTGTTACCTAAAAGAGAACTTCATGGTCAGAATCCTGTTGTAACTCCATGCAATAAACAGTTCCTGAGTCAATTTGAAATGCAGTCCAGGAAAACTACACAATCAGGACAAATGTCTGGGGAAGGTAAAGCTGGTCCTCCAGGAGGCAGTTCCCGTGCAGCATTTCCACAAGGTGGTAGAGGACGGGGCCGTTTTCCAGGGGCTGTTCCTGGTGGGGACAGATTTCCTGGGCCAGCAGGACCAGGAGGGCCACCCCCACCTTTTCCAGGAAATTTGATCAAGCATCTTGTTAAAGGAACTCGGCCTTTGTTCCTGGAAACTAGGATTCCATGGCATATGGGGCACAGCATAGAGGAAATACCCATTTTTGGCCTAAAAGCTGGACAGACTCCACCACGTCCACCCTTAGGTCCTCCAGGCCCACCTGGTCCACCAGGTCCTCCACCTCCTGGTCAGGTTCTGCCTCCTCCTCTAGCTGGGCCTCCTAATCGAGGAGATCGCCCTCCACCACCAGTTCTTTTTCCTGGACAACCTTTTGGGCAGCCTCCATTGGGTCCACTTCCTCCTGGCCCTCCACCTCCAGTTCCAGGCTACGGCCCCCCTCCTGGCCCACCACCTCCACAACAGGGACCACCTCCACCTCCAGGCCCCTTTCCACCTCGTCCACCCGGTCCACTTGGGCCACCCCTTACACTAGCTCCTCCTCCGCATCTTCCTGGACCACCTCCAGGTGCCCCACCGCCAGCTCCGCATGTGAACCCAGCTTTCTTTCCTCCACCAACTAACAGTGGCATGCCTACATCAGATAGCCGAGGTCCACCACCAACAGATCCATATGGGCGACCTCCACCATATGATAGGGGTGACTATGGCCCCCCTGGAAGGGAAATGGATACTGCAAGAACGCCATTGAGTGAAGCTGAATTTGAAGAAATCATGAATAGAAATAGGGCAATCTCAAGCAGTGCTATTTCGAGAGCTGTGTCTGATGCCAGTGCTGGTGATTATGGGAGTGCTATTGAGACACTGGTAACTGCAATTTCTTTAATTAAACAATCCAAAGTATCTGCTGATGATCGTTGCAAAGTTCTTATTAGTTCTTTGCAAGATTGCCTTCATGGAATTGAGTCCAAGTCTTATGGTTCTGGATCAAGACGTGAACGATCAAGAGAGAGGGACCATAGTAGATCACGAGAAAAGAGTCGACGTCATAAATCCCGTAGTAGAGACCGTCATGACGATTATTACAGAGAGAGAAGCAGAGAACGAGAGAGGCACCGGGATCGTGACCGAGACCGTGACCGAGAGCGTGACCGAGAGCGCGAATATCGTCATCGTTAGAAG'
    _seq = seq
    _y = np.zeros(len(seq))
    for i in range(len(label)):
        _y[label[i][0]-1: label[i][1]] = 1.
    _data = []
    for j in range(len(_seq)):
        if j < window_size // 2:
            _data.append((window_size - len(_seq[:j + window_size // 2]) - 1) * '0' + _seq[:j + window_size // 2 + 1])
        elif j >= len(_seq) - window_size // 2:
            _data.append(_seq[j - window_size // 2:] + '0' * (window_size - len(_seq[j - window_size // 2:])))
        else:
            _data.append(_seq[j - window_size // 2: j + window_size // 2 + 1])
    _data = to_one_hot(_data, window_size=window_size)
    # return _data
    return _data, _y


def full_RNA_data(window_size, rbp_name):
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i + 1].strip()
    f_name_seq.close()

    f_p = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_test.fasta".format(rbp_name))
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()

    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq

    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:17]][int(line[-2]) - 1:int(line[-1])] = 1.

    _data = []
    _y = []
    len_seq = []
    rna_name = list(name_y.keys())
    counts, full_rna_names = [0], []
    count = 0
    for i in range(len(rna_name)):
        # if i > 0:
        #     break
        _seq = name_seq[rna_name[i]]
        full_rna_names.append(rna_name[i])
        _y.extend(name_y[rna_name[i]])
        len_seq.append(len(_seq))

        assert len(name_y[rna_name[i]]) == len(name_seq[rna_name[i]])
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
    return to_one_hot(_data, window_size=window_size), np.array(_y), np.array(counts), np.array(full_rna_names)


def t_TSNE(window_size, rbp_name):
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i + 1].strip()
    f_name_seq.close()

    f_p = open("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site/{0}/{0}_test.fasta".format(rbp_name))
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()

    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq

    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:17]][int(line[-2]) - 1:int(line[-1])] = 1.

    _data = []
    _y = []
    len_seq = []
    rna_name = list(name_y.keys())
    counts, full_rna_names = [0], []
    count = 0
    for i in range(len(rna_name)):
        k = 0
        if i<k:
          continue
        elif i > k:
            break
        _seq = name_seq[rna_name[i]]
        print('RBP name:', rna_name[i])
        full_rna_names.append(rna_name[i])
        _y.extend(name_y[rna_name[i]])
        len_seq.append(len(_seq))

        assert len(name_y[rna_name[i]]) == len(name_seq[rna_name[i]])
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
    return to_one_hot(_data, window_size=window_size), np.array(_y), np.array(counts), np.array(full_rna_names)

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


if __name__ == "__main__":
    _window_size, _rbp_name = 257, "AUF1"
    _data_pos, _data_neg = get_train_data(window_size=_window_size, rbp_name=_rbp_name)
    # print(_data_pos)
    print(_data_pos.shape, _data_neg.shape)

