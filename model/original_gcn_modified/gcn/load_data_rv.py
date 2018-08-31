
# coding: utf-8

import numpy as np
import pickle as pkl
import scipy.sparse as sp

def load_data_rv(wd):
    f=open(wd+'adj.pkl','rb')
    adj_coo=pkl.load(f)
    f.close()
    features=np.loadtxt(wd+'feature.txt')
    train_label=np.loadtxt(wd+'nucleus_train_label.txt')
    val_label=np.loadtxt(wd+'nucleus_val_label.txt')
    test_label=np.loadtxt(wd+'nucleus_test_label.txt')
    f=open(wd+'mask.pkl','rb')
    train_mask,val_mask,test_mask=pkl.load(f)
    f.close()
    return adj_coo.toarray(),sp.coo_matrix(features),train_label,val_label,test_label,train_mask,val_mask,test_mask

