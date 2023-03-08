import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from .data_loader import *
from .tools import *
from .metrics import *




def metric_eval(m_dataset,t_dataset,b_size,model_dir,ensemble_num,ece_num_bins,shifts = [0.,0.,0.,0.,0.],num_layers = 0,num_classes=10):
    # check model path validity
    if ensemble_num ==1:
        assert os.path.isfile(model_dir)
    else:
        assert os.path.isdir(model_dir)
    
    # create psudo args for Net initialisation
    p_args = pseudo_args(num_classes=num_classes,num_layers =num_layers)
    # Load model and data
    model = load_model(p_args,m_dataset,model_dir,ensemble_num)
    dataset_test,classes = load_data(t_dataset,rotation = shifts[0], translation = shifts[1], noise = [shifts[2],shifts[3]],flip = shifts[4])

    # create data loader
    dataset = torch.utils.data.DataLoader(dataset_test, batch_size=b_size, shuffle=True, num_workers=2)
    

    # Init 
    num_classes = len(classes)
    correct = 0
    total = 0
    num_bins = 100
    b_score = 0
    ece = 0
    ece_acc_arr = np.zeros(ece_num_bins)
    nll = 0

    aurroc_num_batches = max(int(np.floor(len(dataset_test)/b_size)),1)
    auroc_probs = np.zeros((aurroc_num_batches*b_size,num_classes))
    auroc_labels = np.zeros(aurroc_num_batches*b_size)
    hist_corr = np.zeros(num_bins)
    hist_incorr = np.zeros(num_bins)
    hist_bins = [] 

    # since we're not training, we don't need to calculate the gradients for our output

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            images, label = data
            # get logit for images (currently only softmax)
            sm_out_prob,_ = get_logit(images,model,len(classes),ensemble_num)
            labels = label.clone().detach().numpy().astype(int)
            predicted = np.argmax(sm_out_prob, axis=-1)
            pred_prob= np.max(sm_out_prob, axis=-1)

            #Add to total
            total += len(labels)
            correct += (predicted == labels).sum().item()

            # histogram for confidence vs # of correct/wrong predictions
            correct_preds = pred_prob[labels==predicted]
            incorrect_preds = pred_prob[labels!=predicted]
            pc_hist_count,pc_hist_bins = np.histogram(correct_preds, bins=num_bins, range = (0.,1.))
            pi_hist_count,pi_hist_bins = np.histogram(incorrect_preds, bins=num_bins, range = (0.,1.))
            hist_corr += pc_hist_count
            hist_incorr += pi_hist_count
            hist_bins = pi_hist_bins

            # brier score, ece and nll
            batch_bs = brier_score(sm_out_prob,labels,num_classes)
            batch_ece,batch_ece_acc_arr = expected_calibration_errorscore_bins(sm_out_prob,labels,ece_num_bins)
            batch_nll = negative_log_liklihood(sm_out_prob,labels)

            # nll, and running average of b_score, ece
            b_score = ((total - len(labels))*b_score + len(labels)*batch_bs)/total
            ece = ((total - len(labels))*ece + len(labels)*batch_ece)/total
            ece_acc_arr = ((total - len(labels))*ece_acc_arr + len(labels)*batch_ece_acc_arr)/total
            nll += batch_nll

            if batch_idx < aurroc_num_batches:
                auroc_probs[(batch_idx*b_size):((batch_idx+1)*b_size),:] = sm_out_prob
                auroc_labels[(batch_idx*b_size):((batch_idx+1)*b_size)] = labels

    auroc = auc_roc(auroc_probs,auroc_labels.astype(int),num_classes)
    auroc_label = draw_roc(auroc_probs,auroc_labels.astype(int),num_classes)

    met_dict = {
        "correct" : correct,
        "total" : total,
        "brier_score" : b_score,
        "ece" : ece,
        "ece_acc_arr" : ece_acc_arr,
        "nll" : nll,
        "auroc" : auroc,
        "auroc_label" : auroc_label,
        "hist_corr" : hist_corr,
        "hist_incorr" : hist_incorr,
        "hist_bins" : hist_bins
    }
    return met_dict


def metric_eval_flwr(dataloader,net,device,b_size,ensemble_num,ece_num_bins,shifts = [0.,0.,0.,0.,0.],num_layers = 0,num_classes=10):
    # Init 
    net.to(device)
    net.eval()

    # Accuracy init
    correct = 0
    total = 0

    # B_score init
    b_score = 0

    # ece init
    ece = 0
    ece_acc_arr = np.zeros(ece_num_bins)

    # nll init
    nll = 0

    # Auroc init
    num_bins = 100
    # We are only looking at the first around 4000 samples because auroc runs pretty slow.
    aurroc_num_batches = int(np.ceil(4000/b_size))
    auroc_probs = np.zeros((aurroc_num_batches*b_size,num_classes))
    auroc_labels = np.zeros(aurroc_num_batches*b_size)
    hist_corr = np.zeros(num_bins)
    hist_incorr = np.zeros(num_bins)
    hist_bins = []

    # loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0

    # since we're not training, we don't need to calculate the gradients for our output

    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(dataloader):
            images, label = images.to(device), label.to(device)
            # get logit for images (currently only softmax)
            sm_out_prob, outputs = get_logit(images,net,num_classes,ensemble_num)
            loss += criterion(outputs,label).item()
            labels = label.clone().detach().cpu().numpy().astype(int)
            predicted = np.argmax(sm_out_prob, axis=-1)
            pred_prob= np.max(sm_out_prob, axis=-1)


            #Add to total
            total += len(labels)
            correct += (predicted == labels).sum().item()

            # histogram for confidence vs # of correct/wrong predictions
            correct_preds = pred_prob[labels==predicted]
            incorrect_preds = pred_prob[labels!=predicted]
            pc_hist_count,pc_hist_bins = np.histogram(correct_preds, bins=num_bins, range = (0.,1.))
            pi_hist_count,pi_hist_bins = np.histogram(incorrect_preds, bins=num_bins, range = (0.,1.))
            hist_corr += pc_hist_count
            hist_incorr += pi_hist_count
            hist_bins = pi_hist_bins

            # brier score, ece and nll
            batch_bs = brier_score(sm_out_prob,labels,num_classes)
            batch_ece,batch_ece_acc_arr = expected_calibration_errorscore_bins(sm_out_prob,labels,ece_num_bins)
            batch_nll = negative_log_liklihood(sm_out_prob,labels)

            # nll, and running average of b_score, ece
            b_score = ((total - len(labels))*b_score + len(labels)*batch_bs)/total
            ece = ((total - len(labels))*ece + len(labels)*batch_ece)/total
            ece_acc_arr = ((total - len(labels))*ece_acc_arr + len(labels)*batch_ece_acc_arr)/total
            nll += batch_nll

            # This thing seem to be unable to cope with changing bs. 
            if batch_idx < aurroc_num_batches:
                auroc_probs[(batch_idx*b_size):((batch_idx+1)*b_size),:] = sm_out_prob
                auroc_labels[(batch_idx*b_size):((batch_idx+1)*b_size)] = labels

    auroc = auc_roc(auroc_probs,auroc_labels.astype(int),num_classes)
    auroc_label = draw_roc(auroc_probs,auroc_labels.astype(int),num_classes)
    met_dict = {
        "correct" : correct,
        "total" : total,
        "brier_score" : b_score,
        "ece" : ece,
        "ece_acc_arr" : ece_acc_arr,
        "nll" : nll,
        "auroc" : auroc,
        "auroc_label" : auroc_label,
        "hist_corr" : hist_corr,
        "hist_incorr" : hist_incorr,
        "hist_bins" : hist_bins,
        "accuracy" : float(correct)/float(total),
        "loss" : loss
    }
    return met_dict