import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpn import GPN
import time
import pickle
from tqdm import tqdm


# args
parser = argparse.ArgumentParser(description="GPN test")
parser.add_argument('--size', default=50, help="size of model")
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--test_size', default=50, help="size of TSP")
parser.add_argument('--test_steps', default=100, help='')
args = vars(parser.parse_args())


start = 5
end = 100
step = 5

result_list_cost = []
for p in tqdm(range(start,end + step, step)):
    B = int(args['batch_size'])
    n_test = int(args['test_steps'])
    test_size = p
    B = 250
    n_test = 100
    load_data = '../../attention-learn-to-route/data/tsp/tsp{}_uniform_test_seed1234.pkl'.format(p)
    with open(load_data, 'rb') as f:
        test_data = pickle.load(f)

    test_data = np.array(test_data)
    print(test_data.shape)
    dataset_size = test_data.shape[0]
    print(dataset_size/B)

    batch_num = int(dataset_size/B)

    test_data = torch.from_numpy(test_data)
    Z = test_data.view(batch_num, B, test_size, 2)
    Z = Z.to(torch.float32)
    load_root ='./model/gpn_tsp/tsp_50__220523_16_49/'+'epoch{}.pt'.format(99)
    # load_root ='/home/lhz_models/graph-pointer-network/tsp_larger/model/gpn_tsp500/'+'epoch{}.pt'.format(9)

    print('=========================')
    print('prepare to test')
    print('=========================')
    print('Hyperparameters:')
    print('batch size', B)
    print('test size', test_size)
    print('test steps', n_test)
    print('load root:', load_root)
    print('=========================')
        
    # greedy
    model = torch.load(load_root).cuda()

    tour_len = 0
    total_len = 0

    start_time = time.time()

    for m in range(batch_num):
        tour_len = 0
        
        X = Z[m].cuda()
        
        mask = torch.zeros(B,test_size).cuda()
        
        R = 0
        Idx = []
        reward = 0
        
        Y = X.view(B,test_size,2)           # to the same batch size
        x = Y[:,0,:]
        h = None
        c = None
        
        for k in range(test_size):
            
            output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            
            idx = torch.argmax(output, dim=1)
            Idx.append(idx.data)
            
            Y1 = Y[[i for i in range(B)], idx.data].clone()
            if k == 0:
                Y_ini = Y1.clone()
            if k > 0:
                reward = torch.norm(Y1-Y0, dim=1)
        
            Y0 = Y1.clone()
            x = Y[[i for i in range(B)], idx.data].clone()
            
            R += reward

            mask[[i for i in range(B)], idx.data] += -np.inf
            
        R += torch.norm(Y1-Y_ini, dim=1)

        tour_len += R.mean().item()

        print('test:{}, total length:{}'.format(m, tour_len))
        
        total_len += tour_len

    print('total tour length:', total_len/batch_num)
    result_list_cost.append(total_len/batch_num)
    print(time.time() - start_time)
print(result_list_cost)
