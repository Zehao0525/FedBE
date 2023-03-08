#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import pdb
import os
import pickle

import numpy as np
from swag import SWAG_server

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.sampling import *
from utils.options import args_parser
from utils.tools import *
from utils.main_extensions import *

from eval_utils.metrics_evaluator import metric_eval

from models.Update import SWAGLocalUpdate, ServerUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, create_local_init
from models.FedM import FedAvgM
from models.test import test_img
import resnet

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    # make all the directories
    args.log_dir = os.path.join(args.log_dir)   
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)  

    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            print (arg, getattr(args, arg), file=f)

    args.acc_dir = os.path.join(args.log_dir, "acc")
    if not os.path.exists(args.acc_dir):
        os.makedirs(args.acc_dir)  
        
    model_dir = os.path.join(args.log_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)         

    # transform train parameters
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # load dataset and split users
    if args.dataset == 'mnist':
        args.num_classes = 10
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_eval = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)        
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, server_id, cnts_dict  = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        args.num_classes = 10
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_val)
        dataset_eval = datasets.CIFAR10('./data/cifar', train=True, transform=transform_val, target_transform=None, download=True)
        if args.iid:
            dict_users, server_id = cifar_iid(dataset_train, args.num_users, num_data=args.num_data)
        else:
            dict_users, server_id, cnts_dict = cifar_noniid(dataset_train, args.num_users, num_data=args.num_data, method=args.split_method)
    else:
        exit('Error: unrecognized dataset')
        
    train_ids = set()
    # dict_users.items() is the content of the dictionary
    for u,v in dict_users.items():
        train_ids.update(v)
    # train_ids is the liats of all the ids in dict_users for all the users in a 1d array
    train_ids = list(train_ids)     

    img_size = dataset_train[0][0].shape
    # build model
    # models stored in models.Nets
    if args.model == 'cnn' and 'cifar' in args.dataset:
        net_glob = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    elif "resnet" in args.model and 'cifar' in args.dataset:
        net_glob = resnet.resnet32(num_classes=args.num_classes)   
    else:
        exit('Error: unrecognized model')    
        
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_local_list = []
    loss_local_test_list = []
    entropy_list = []
    cv_loss, cv_acc = [], []
    

    acc_local_list = []
    acc_local_test_list = []
    acc_local_val_list = []
    
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    net_glob.apply(weights_init)    

    # Arguments:
    # q: mp.Manager.Queue
    # device_id: the gpu thread being used to train the client
    # net_glob: deep copy of net_glob
    # iters: Current round number
    # idx: idx of the client participating in the current round (range(m))
    # val_id: server_id
    # generator: None
    #
    # return: 
    # A trained teacher model and its index (also put on manger queue)
    def client_train(q, device_id, net_glob, iters, idx, val_id=server_id, generator=None):
        device=torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        print("device used:",device)
        # LearningRate schedule (def in "tools")
        lr = lr_schedule(args.lr, iters, args.rounds)  

        # local_sch : either step or adaptive. 
        if args.local_sch == "adaptive":
            # local_ep : the number of local epochs
            # adaptive scheduler (def in "tools")
            running_ep = adaptive_schedule(args.local_ep, args.epochs, iters, args.adap_ep)
        if running_ep != args.local_ep:
            print("Using adaptive scheduling, local ep = %d."%args.adap_ep)
        else:
            running_ep = args.local_ep

        # In models
        local = SWAGLocalUpdate(args=args, 
                                device=device, 
                                dataset=dataset_train, 
                                idxs=dict_users[idx], 
                                server_ids=val_id, 
                                test=(dataset_test, range(len(dataset_test))), 
                                num_per_cls=cnts_dict[idx]   )

        # train a model using SWAGLocalUpdate, this model is called teacher
        teacher = local.train(net=net_glob.to(device), running_ep=running_ep, lr=lr)
        q.put([teacher, idx])
        return [teacher, idx]

    # Arguments:
    # q : mp.Manager.Queue()
    # device_id : a constant. pretty sure this one doesn't do anything.
    # net_glob : global deep network
    # teachers: set of sampled teachers (and maybe also clients)
    # global_ep: current round number
    # w_org : None
    # base_teachers = None
    #
    # !!!!!!!!!Output:
    # w_swa : weight after stocastic weight averaging
    # w_glob : 
    # train_acc, val_acc, test_acc : 
    # loss, entropy : 
    def server_train(q, device_id, net_glob, teachers, global_ep, w_org=None, base_teachers=None):
        device=torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() and args.num_gpu != -1 else 'cpu')
        student = ServerUpdate(args=args, 
                            device=device, 
                            dataset=dataset_eval, 
                            server_dataset=dataset_eval, 
                            server_idxs=server_id, 
                            train_idx=train_ids, 
                            test=(dataset_test, range(len(dataset_test))), 
                            w_org=w_org, 
                            base_teachers=base_teachers)
      
        w_swa, w_glob, train_acc, val_acc, test_acc, loss, entropy = student.train(net_glob, teachers, args.log_dir, global_ep)

        q.put([w_swa, w_glob, train_acc, val_acc, test_acc, entropy])
        return [w_swa, w_glob, train_acc, val_acc, test_acc, entropy]
      
    # Arguments:
    # q : mp.Manager.Queue()
    # net_glob : blobal deep network
    # dataset : dataset being tested
    # ids : a list of indexes of the data from the dataset being tested
    #
    # Output:
    # [acc, loss] : accuracy and loss of the model (also put on Queue)
    def test_thread(q, net_glob, dataset, ids):
        # acc: accuracy
        # loss: test loss
        acc, loss = test_img(net_glob, dataset, args, ids, cls_num=args.num_classes)
        q.put([acc, loss])
        return [acc, loss]

    def eval(net_glob, tag='', server_id=None):
        # testing
        q = mp.Manager().Queue()

        p_tr = mp.Process(target=test_thread, args=(q, net_glob, dataset_eval, train_ids))  
        p_tr.start()
        p_tr.join()
        [acc_train, loss_train] = q.get()

        q2 = mp.Manager().Queue()
        p_te = mp.Process(target=test_thread, args=(q2, net_glob, dataset_test, range(len(dataset_test))))  
        p_te.start()
        p_te.join()

        [acc_test,  loss_test] = q2.get()

        q3 = mp.Manager().Queue()
        p_val = mp.Process(target=test_thread, args=(q3, net_glob, dataset_eval, server_id))
        p_val.start()
        p_val.join()

        [acc_val,  loss_val] = q3.get()

        print(tag, "Training accuracy: {:.2f}".format(acc_train))
        print(tag, "Server accuracy: {:.2f}".format(acc_val))
        print(tag, "Testing accuracy: {:.2f}".format(acc_test))

        del q
        del q2 
        del q3 

        return [acc_train, loss_train], [acc_test,  loss_test], [acc_val,  loss_val]
    

    def put_log(logger, net_glob, tag, iters=-1):
        [acc_train, loss_train], [acc_test,  loss_test], [acc_val,  loss_val] = eval(net_glob, tag=tag, server_id=server_id)

        if iters==0:
            open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "w")
            open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "w")
            open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "w")
            open(os.path.join(args.acc_dir, tag+"_test_loss.txt"), "w")

        with open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, acc_train))
        with open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, acc_test))
        with open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, acc_val))          
        with open(os.path.join(args.acc_dir, tag+"_test_loss.txt"), "a") as f:
            f.write("%d %f\n"%(iters, loss_test))
          
        if "SWA" not in tag:
            logger.loss_train_list.append(loss_train)
            logger.train_acc_list.append(acc_train)

            logger.loss_test_list.append(loss_test)
            logger.test_acc_list.append(acc_test)

            logger.loss_val_list.append(loss_val)
            logger.val_acc_list.append(acc_val)
        else:
            if tag =="SWAG":
                logger.swag_train_acc_list.append(acc_train)
                logger.swag_val_acc_list.append(acc_val) 
                logger.swag_test_acc_list.append(acc_test)            
            else:
                logger.swa_train_acc_list.append(acc_train)
                logger.swa_val_acc_list.append(acc_val) 
                logger.swa_test_acc_list.append(acc_test)   
    

    def put_oracle_log(logger, ens_train_acc, ens_val_acc, ens_test_acc, iters=-1):    
        if iters>=0 and iters%args.log_ep!= 0:
            return
        logger.ens_train_acc_list.append(ens_train_acc)
        logger.ens_test_acc_list.append(ens_test_acc)
        logger.ens_val_acc_list.append(ens_val_acc)

        tag = "ens"
        if iters==0:
            open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "w")
            open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "w")
            open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "w")
        
        with open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, ens_train_acc))
        with open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, ens_test_acc))
        with open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "a") as f:
            f.write("%d %f\n"%(iters, ens_val_acc))     


    dist_logger = logger("DIST")
    fedavg_logger = logger("FedAvg")  
    work_tag = args.update    
    

    teachers = [[] for i in range(args.num_users)]  
    generator = None
    best_acc = 0.0
    
    # cnts_dict : count of each label in each clients
    size_arr = [np.sum(cnts_dict[i]) for i in range(args.num_users)]  

    for iters in range(args.rounds):
        print("Server Epouch:",iters)
        w_glob_org = copy.deepcopy(net_glob.state_dict())
        
        net_glob.train()
        loss_locals = []
        acc_locals = []
        acc_locals_test = []
        loss_locals_test = []
        acc_locals_val = []
        
        # m : number of clients participating in this round of training
        m = max(int(args.frac * args.num_users), 1)
        # randomly chose m users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        clients = [[] for i in range(args.num_users)]
        
        num_threads = 5
        for i in range(0, m, num_threads):
            processes = []
            torch.cuda.empty_cache()
            q = mp.Manager().Queue()
            
            for idx in idxs_users[i:i+num_threads]:
                # create process p and append to processes
                p = mp.Process(target=client_train, args=(q, idx%(args.num_gpu), copy.deepcopy(net_glob), iters, idx, server_id, generator))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            while not q.empty(): 
                # fake_out : A trained client model and its index (client number)
                fake_out = q.get()
                idx = int(fake_out[-1])
                clients[idx].append(fake_out[0])

        # Exclude the clients without a model (and also take first model of clients with multiple models)
        clients = [c[0] for c in clients if len(c)>0]
        # clinet_w : a list of the current weights and biases of the variouse layes of the network
        # Each state dict is a dictionary with each entry being "layer_name : layer_tensor_values"
        client_w = [c.state_dict() for c in clients]
        

        if args.store_model and (iters%args.log_ep==0 or iters==args.rounds-1):
            # iter : current round number
            # model_dir : os.path.join(args.log_dir, "models")
            # w_glob_org : (weight of) global model before training
            # client_w : weight of global model after training
            store_model(iters, model_dir, w_glob_org, client_w)
          
        if args.fedM and iters > 1:
            # FedAvg with momentum
            w_glob_avg, momentum = FedAvgM(client_w, args.num_gpu-1, (w_glob_org, momentum), args.mom, size_arr=size_arr)   
        else:
            # Calculate federated average weight and the momentum
            w_glob_avg = FedAvg(client_w, args.num_gpu-1, size_arr=size_arr)
            momentum = {k:w_glob_org[k]-w_glob_avg[k] for k in w_glob_avg.keys()}
        
        # Update the global network
        net_glob.load_state_dict(w_glob_avg)     
        
        if iters%args.log_ep== 0 or iters == args.rounds-1:
            put_log(fedavg_logger, net_glob, tag='FedAvg', iters=iters)
        
        # Generate Teachers
        # Two modes for base teachers: SWAG and FedAvg 
        teachers_list = []
        
        if not args.dont_add_fedavg:
            print("add FedAvg to teachers")
            teachers_list.append(copy.deepcopy(net_glob)) # Add FedAvg

        if args.teacher_type=="SWAG" and iters > args.warmup_ep:
            for i in range(args.num_sample_teacher):
                base_teachers = client_w
                # args : input arguments
                # w_glob_org : orginal global weight. This is a dictionary
                # w_glob_avg : global average weight. Obtained by using fedavg with momentum
                swag_model = SWAG_server(args, w_glob_org, avg_model=w_glob_avg, concentrate_num=1, size_arr=size_arr)
                # Sample models using gaussian methods
                w_swag = swag_model.construct_models(base_teachers, mode=args.sample_teacher) 
                net_glob.load_state_dict(w_swag)
                teachers_list.append(copy.deepcopy(net_glob))  
        else:
            base_teachers = client_w
            print("Warming up, using DIST.")
        
        if args.store_model_pipe and (iters%args.log_ep==0 or iters==args.rounds-1):
            swag_ensemble_sample_w = copy.deepcopy(teachers_list)
        
        if args.use_client:
            teachers_list+=clients          
          
        # Load weights for server training
        net_glob.load_state_dict(w_glob_avg)
        print("Initialize with FedAvg for server training ...")
        # update global weights
        q = mp.Manager().Queue()
        print("Server training...")

        p = mp.Process(target=server_train, args=(q, args.num_gpu-1, net_glob, teachers_list, iters))
        p.start()
        p.join()
        
        [w_glob_mean, w_glob, ens_train_acc, ens_val_acc, ens_test_acc, entropy] = q.get()
        del q 
        
        if best_acc < ens_test_acc:
            best_acc = ens_test_acc
        
        if iters%args.log_ep== 0 or iters == args.rounds-1:
            net_glob.load_state_dict(w_glob_mean)
            put_log(dist_logger, net_glob, tag='DIST-SWA', iters=iters) 
                  
            net_glob.load_state_dict(w_glob)
            put_log(dist_logger, net_glob, tag='DIST', iters=iters) 
            put_oracle_log(dist_logger, ens_train_acc, ens_val_acc, ens_test_acc, iters=iters)        

        if args.update=='FedAvg':  
            net_glob.load_state_dict(w_glob_avg)
            print("Sending back FedAvg!")         
        else:
            if args.use_SWA:
                net_glob.load_state_dict(w_glob_mean)
                print("Sending back student w/ SWA!")
            else:
                net_glob.load_state_dict(w_glob)
                print("Sending back student w/o SWA!")
        
        if args.store_model and iters == args.rounds-1:
            store_model(iters, model_dir, w_glob_org, client_w)
            print("best_acc",best_acc)
        
        # store teacher models, AVG model, w/oSWAG model
        if args.store_model_pipe and (iters%args.log_ep==0 or iters==args.rounds-1):
            if not args.dont_add_fedavg:
                store_teacher_w = swag_ensemble_sample_w[1:]
            else:
                store_teacher_w = swag_ensemble_sample_w
            teacher_w = [t.state_dict() for t in store_teacher_w]
            print(len(teacher_w))
            store_model_2(iters,model_dir,teacher_w,tag='w_swag_teacher_sample')
            store_model_2(iters,model_dir,w_glob_avg,tag='w_client_avg')   
            store_model_2(iters,model_dir,w_glob,tag='w_woSWA')  

        del clients
    net_glob.load_state_dict(w_glob_mean)
    torch.save(net_glob.state_dict(), os.path.join(args.log_dir, "FedBE_wSWA_model"))
    net_glob.load_state_dict(w_glob)
    torch.save(net_glob.state_dict(), os.path.join(args.log_dir, "FedBE_woSWA_model"))
