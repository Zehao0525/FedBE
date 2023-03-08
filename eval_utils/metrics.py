import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from .data_loader import *
from .tools import *

def brier_score(probs,label,num_classes):
        prob = probs
        labels = label.astype(int)
        oh_labels = np.zeros((labels.size, num_classes))
        oh_labels[np.arange(labels.size), labels] = 1
        #label_probs = prob[np.arange(len(labels)),labels]
        #print(np.sum(1 - 2*label_probs + np.square(prob).sum(axis = -1))/len(probs))
        score = np.sum(np.square(prob - oh_labels))/len(probs)
        return score

def expected_calibration_errorscore_bins(prob,label,num_bins,graph=False):
    labels = label.astype(int)
    label_probs = prob[np.arange(labels.size),labels]
    preds = np.argmax(prob,axis = -1)
    pred_probs = prob[np.arange(preds.size),preds]
    binwidth = 1/num_bins
    truth = preds == labels
    ece_conf_arr = np.zeros(num_bins)
    ece = 0
    for i in range(num_bins):
        in__id = np.where((pred_probs>=(binwidth*i)) & (pred_probs<=(binwidth*(i+1))))
        in_id_size = np.size(in__id)
        if in_id_size==0: 
            ece_bin = 0
            ece_conf_arr[i] = 0
            continue
        acc = truth[in__id].sum()/ in_id_size
        conf = pred_probs[in__id].sum()/in_id_size
        ece_bin = (in_id_size/np.size(labels))*np.absolute(acc - conf)
        ece += ece_bin
        ece_conf_arr[i] = acc
    return ece,ece_conf_arr

def negative_log_liklihood(prob,label):
    labels = label.astype(int)
    label_probs = prob[np.arange(len(labels)),labels]
    log_like = np.log2(label_probs)
    return -log_like.sum()

def auc_roc(prob,label,num_classes):
    oh_label = np.zeros((len(label), num_classes))
    oh_label[np.arange(len(label)), label] = 1
    auroc = roc_auc_score(oh_label,prob,multi_class='ovo',average='micro')
    return auroc

def draw_roc(prob,label,num_class):
    auroc = np.zeros(num_class)
    #ftpr = np.zeros((2,num_class,len(label)))
    for i in range(num_class):
        lab_prob = prob[:,i]
        is_classi = label == i
        fpr,tpr,threshold = roc_curve(is_classi,lab_prob)
        #ftpr[0][i] = fpr
        #ftpr[1][i] = tpr
        auroc[i] = roc_auc_score(is_classi,lab_prob)

    return auroc#,ftpr

def pred_entropy(prob):
    entropy = prob*np.log2(prob)
    return entropy

def draw_ece(ece_arr,num_bins):
    accs = (np.arange(num_bins)+0.5)/num_bins
    indices = (np.arange(num_bins))/num_bins
    plt.clf()
    plt.bar(indices, ece_arr, width = 1/num_bins, color='b', label='Output')
    plt.bar(indices, accs, width = 1/num_bins, color='r', alpha=0.5, label='Gap')
    plt.xticks(indices, indices )
    plt.legend()
    plt.savefig("ece_test.png")




class pseudo_args:
    def __init__(self, num_classes=10, num_layers=0):
        self.num_classes = num_classes
        self.num_layers = num_layers


