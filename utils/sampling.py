#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import pdb
from torchvision import datasets, transforms
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, num_data=60000):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # dict_users = {0：array([], dtype=int64), 1: array([], dtype=int64) ....}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()[:num_shards*num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # Remove rand_set from idxshard
        for rand in rand_set:
            add_idx = np.array(list(set(idxs[rand*num_imgs:(rand+1)*num_imgs]) ))
            dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)

    cnts_dict = {}
    with open("mnist_%d_u%d.txt"%(num_data, num_users), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
    
    server_idx = list(range(num_shards*num_imgs, 60000))
    return dict_users, server_idx, cnts_dict
    # dict_users: Set of size num_users each containing set of data for each user
    # server_idx: a list of indexes beyween numb_shard * num_img and 60000
    # cnt_dict: number of each labaels in each client

def cifar_iid(dataset, num_users, num_data=50000):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param num_data: number of data distributed to clients
    :return: dict of image index
    """
    server_idx = np.array([],dtype = "int64")
    labels = np.array(dataset.targets)
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    if num_data < 50000:
      server_idx = np.random.choice(all_idxs, 50000-num_data, replace=False)
      all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)
    
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False).astype(int)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))

    cnts_dict = {}
    for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts 

    return dict_users, server_idx, cnts_dict
    
def cifar_noniid(dataset, num_users, num_data=50000, method="step",step_qs_p = 0.1):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # dataset: datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    # labels is a set of integers marking the label of the data in the dataset
    labels = np.array(dataset.targets)
    _lst_sample = 10 
    
    if method=="step":
      
      num_shards = num_users*2
      num_imgs = 50000// num_shards # // is divide floor
      idx_shard = [i for i in range(num_shards)] 
      
      idxs = np.arange(num_shards*num_imgs) # 50000 - 50000 mod (num_users*2)
      # sort labels
      idxs_labels = np.vstack((idxs, labels[:len(idxs)]))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      # iterating over classes
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      # We end with (10)
      least_idx = np.reshape(least_idx, (num_users, -1))
      # least_idx have shape (num_users,_lst_sample*10)
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          # rand_set is chose 2 from num_users*2
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, (rand+1)*num_imgs))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              

              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)

    elif method=="qs_step":
      if step_qs_p>0.5: step_qs_p = 0.5
      num_shards = num_users
      num_imgs = 50000// num_shards # // is divide floor
      idx_shard = [i for i in range(num_shards)] 
      
      idxs = np.arange(num_shards*num_imgs) # 50000 - 50000 mod (num_users*2)
      # sort labels
      idxs_labels = np.vstack((idxs, labels[:len(idxs)]))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=int)
      # iterating over classes
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      # We end with (10)
      least_idx = np.reshape(least_idx, (num_users, -1))
      # least_idx have shape (num_users,_lst_sample*10)
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      

      # divide and assign
      dict_users = { i: np.array([], dtype='int64') for i in range(num_users)}

      def popoff(list_ind,index):
          return_val = list_ind[index][0]
          list_ind[index] = list_ind[index][1:]
          if len(list_ind[index]) == 0:
              del list_ind[index]
          return return_val

      # This mechanism will let users with a lower index to generally get more data than users higher indexes
      # Our users are identical in every way except for the data they carry.
      list_data = []
      for i in range(num_users):
          ransam = int(np.random.uniform(step_qs_p,1-step_qs_p)*num_imgs)
          ld_temp = np.array([[i*num_imgs,i*num_imgs+ransam],[i*num_imgs+ransam,(i+1)*num_imgs]], dtype='int64')
          list_data.append(ld_temp)
      for i in range(num_users):
          # rand_set is chose 2 from num_users*2
          idx_range = np.zeros((2,2), dtype='int64')
          for j in range(2):
            rand_ind = np.random.choice(np.arange(len(list_data)))
            idx_range[j] = popoff(list_data,rand_ind)
          for ir in idx_range:
              idx_i = list(set(range(ir[0], ir[1])) )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              

              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
 
    elif method == "dir":
      min_size = 0
      K = 10
      y_train = labels
      
      _lst_sample = 2

      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      local_idx = np.array([i for i in range(50000) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      net_dataidx_map = {}
      # {} defines a dictionary. {0:data,1:data...}
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

      while min_size < 10:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id in local_idx]
              
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(0.1, num_users))
              ## Balance
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              min_size = min([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        # 
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
    # dict_users: Set of size num_users each containing the indexes set of data for each user
    # server_idx: a list of indexes beyween numb_shard * num_img and 60000
    # cnt_dict: number of each labaels in each dict
    return dict_users, server_idx, cnts_dict



def cifar_test(dataset, num_users, num_data=10000,cnts_dict = None):
    if cnts_dict==None:
        print("cont_dict = None")
        cnts_dict = {i: np.zeros(10, dtype='int64')+(40000/(num_users*10)) for i in range(num_users)}
    cnt_sum = np.sum([np.sum(cnts_dict[i]) for i in range(len(cnts_dict))])
    shrink_fact = cnt_sum/num_data
    shrink_fact = 1.1*shrink_fact

    # Extract the test set
    labels = np.array(dataset.targets)
    
    idxs = np.arange(num_data) # 50000 - 50000 mod (num_users*2)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    #idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idx_by_labels = []
    for i in range(10):
        ibl = idxs_labels[0][idxs_labels[1,:]==i]
        #ibl = ibl[0,:]
        np.random.shuffle(ibl)
        idx_by_labels.append(ibl)
    
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(num_users):
        for j in range(10):
            num_d_j = int(cnts_dict[i][j]/shrink_fact)
            add_idx = idx_by_labels[j][:num_d_j]
            idx_by_labels[j] = idx_by_labels[j][num_d_j:]
            test_dict_users[i] = np.concatenate((test_dict_users[i], add_idx), axis=0) 
    
    test_cnts_dict = {}
    with open("test_data_%d_u%d.txt"%(num_data, num_users), 'w') as f:
      for i in range(num_users):
        labels_i = labels[test_dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        test_cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) )) 

    return test_dict_users, test_cnts_dict
