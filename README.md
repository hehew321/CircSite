## Overview
Circular RNAs (circRNAs) interact with RNA-binding proteins (RBPs) to modulate gene expression. To date, most computational methods for predicting RBP binding sites on circRNAs focus on circRNA fragments instead of full-length circRNAs. These methods detect whether an circRNA fragment contains a binding site, but cannot determine where is the binding site and how many binding sites on the whole circRNA. We report a hybrid deep learning-based tool, called CircSite, to predict RBP binding sites at single-nucleotide resolution and detect key contributed sequence contents on full-length circRNAs. CircSite takes advantages of convolutional neural network (CNN) and Transformer for learning local and global representations, respectively. We construct 37 datasets for RBP-binding full-length circRNAs and the experimental results show that CircSite offers accurate predictions of RBP binding nucleotides and detects known binding motifs. To the best of our knowledge, CircSite is the first computational tool to explore the binding nucleotides of RBPs on full-length circRNAs.
## Dependency

* python3.6

* matplotlib 3.1.3

* torch 1.8.1

* skearn 0.20.0

* numpy 1.19.3
## Dataset
* Benchmark datasets 
In this study, we construct a benchmark dataset of RBP-binding full-length circRNAs for 37 RBPs, it consists of training and test set at the nucleotide level. In addition, we construct an independent test set consisting of full-length circRNAs for 37 RBPs, this set is used to evaluate the performance of RBPsutie 2.0 on predicting whether the full-length circRNA can interact with a given RBP.
### Nucleotide-level training and test set[(Click here to download)](http://www.csbio.sjtu.edu.cn/bioinf/CircSite/circ_dataset/nucleotide-level_dataset.zip)
We first construct benchmark datasets of RBP binding sites on full-length circRNAs. Over 120,000 full-length circRNAs sequences for 37 RBPs are extracted from the circRNA interactome database (https://circinteractome.nia.nih.gov/). For each RBP, we first split binding full-length sequences into training and test set with a ratio 8:2. Considering that high sequence similarity in the training and test sets may lead to overestimated performance, we use CD-HIT with a similarity threshold of 0.8 to remove redundant sequences in the test sets. In addition, to avoid potential bias caused by sequences that were too short and too long, we only keep those sequences with a length between 200 and 6000, where the number of sequences within this interval takes over 90% of the total full-length sequences. 
### Independent full-length circRNA test set[(Click here to download)](http://www.csbio.sjtu.edu.cn/bioinf/CircSite/circ_dataset/independent_test__dataset.zip)
To better explore the advantages of RBPsuite 2.0, we try to predict whether the full-length circRNA can interact with a given RBP or not. A total of 37 datasets for 37 RBPs are collected, each dataset set consists of 100 positive and negative full-length circRNAs randomly selected from circinteractome database. The positive samples were composed of full-length circRNAs that have binding sites for a given RBP and do not appear in the training set. The negative samples were composed of full-length circRNAs that do not have any binding sites with the given RBP. In order to make a objective evaluation, we use CD-HIT with a similarity threshold of 0.8 to remove redundant sequences in the independent test set to the training set. In this independent test set, each sample is a full-length circRNA.


## Contact
* 2008xypan@sjtu.edu.cn

## Reference
* xxxxx

