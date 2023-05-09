import os
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
#from ..models.Nets import MLP, CNNMnist, CNNCifar
from models.Nets import MLP, CNNMnist, CNNCifar
import torch.nn.functional as F
import numpy as np
import pickle


def load_model(args,dataset,model_dir,ensemble_num):
    if ensemble_num>1:
        model = []
        for m in range(ensemble_num):
            if dataset == 'cifar10':
                modelm = CNNCifar(args=args)
            elif dataset == 'mnist':
                modelm = CNNMnist(args=args)
            elif dataset == 'mlp':    
                modelm = MLP(args=args)
                exit('Error: unrecognized dataset')
            else:
                exit('Error: unrecognized dataset')
            modelm_dir = os.path.join(model_dir,args.model_name+"_"+str(m))
            print("modelm_dir:",modelm_dir)
            assert os.path.isfile(modelm_dir)
            modelm.load_state_dict(torch.load(modelm_dir))
            modelm.eval()
            model.append(copy.deepcopy(modelm))
    else:
        if dataset == 'cifar10':
            model = CNNCifar(args=args)
        elif dataset == 'mnist':
            model = CNNMnist(args=args)
        elif dataset == 'mlp':    
            model = MLP(args=args)
            exit('Error: unrecognized dataset')
        else:
            exit('Error: unrecognized dataset')
        model.load_state_dict(torch.load(model_dir))
        model.eval()
    return model


def get_logit(images,model,num_classes,ensemble_num):
    if ensemble_num>1:
        sm_out_probs = np.zeros((ensemble_num,len(images),num_classes))
        sum_outputs = np.zeros((ensemble_num,len(images),num_classes))
        for m in range(ensemble_num):
            outputs = model[m](images)
            sum_outputs+= outputs
            sop = F.softmax(outputs,dim=1).detach().clone()
            sop = sop.cpu().numpy()
            sm_out_probs[m] = sop
        sm_out_prob = np.mean(sm_out_probs,axis=0)
        outputs = np.mean(sum_outputs, axis = 0)
    else:
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        sm_out_prob = F.softmax(outputs,dim=1).detach().clone().cpu().numpy()
    
    return sm_out_prob, outputs

def save_metrics(met_dict,file_dir,tag = ''):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir) 
    save_dir = os.path.join(file_dir,tag+'.pickle')  
    with open(save_dir, 'wb') as filehandler:
        pickle.dump(met_dict, filehandler,protocol=pickle.HIGHEST_PROTOCOL)

def load_metrics(file_dir):
    with open(file_dir, 'rb') as filehandler:
        met_dict = pickle.load(filehandler)
    return met_dict
