import math
import pickle
import sys
sys.path.append("..")
import copy
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils.functions import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from tqdm import tqdm
mp = torch.multiprocessing.get_context('spawn')


def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError


def test_attention_choose(opts):
    data_path = "../data/tsp/tsp50_validation_uniform_seed4321.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    opts = parser.parse_args()
    model, _ = load_model(opts.model)
    model.decode_type = "sampling"
    for i in tqdm(range(1000)):
        cost, log_likelihood = model(torch.Tensor(data[i]).unsqueeze(0))


def load_attention_choose_data():
    save_path = "/home/lhz_models/python_shell/results/tsp/AM_attention_choose/"
    # file_path = save_path + "tsp_50_rollout_cluster_linear_300_layer_{}.pkl"
    file_path = save_path + "1000_tsp50_rollout_mix_lc_layer_1000_{}.pkl"
    mulitiple_head_index_layer = np.zeros((3,8,100))
    mulitiple_head_mean_distance  = np.zeros((3,8))
    for i in tqdm(range(3000)):
        index_layer = i % 3
        with open(file_path.format(i), 'rb') as f:
            attn = pickle.load(f)
            # print(pickle.load(f))
        data_path = "../data/tsp/tsp50_validation_uniform_seed4321.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        data = np.array(data[0])
        print(data.shape)
        dist = np.zeros((50, 50))
        for k in range(50):
            for p in range(50):
                dist[k][p] = get_dist(data[k], data[p])
        dist_index = np.argsort(dist)
        for j in range(8):
            mean_distance_total = 0
            for k in range(50):
                current_index = np.argmax(attn[j,0,k])
                mulitiple_head_index_layer[index_layer,j,dist_index[k][current_index]] += 1
                current_attention = attn[j][0][k]
                mean_distance = current_attention * dist_index[k]
                mean_distance_total += np.sum(mean_distance)
            mulitiple_head_mean_distance[index_layer][j] += mean_distance_total/50
    for i in range(3):
        for j in range(8):
            mulitiple_head_mean_distance[i][j] /= 1000
    with open("/home/lhz_models/attention-learn-to-route/mydata/tsp_50_rollout_cluster_linear_300_attention_choose.pkl", 'wb') as f:
        pickle.dump(mulitiple_head_index_layer, f, pickle.HIGHEST_PROTOCOL)
    print(mulitiple_head_mean_distance.tolist())


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    opts = parser.parse_args()
    # test_attention_choose(opts)
    load_attention_choose_data()




