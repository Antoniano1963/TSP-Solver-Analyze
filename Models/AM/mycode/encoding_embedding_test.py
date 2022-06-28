import math
import pickle
import copy
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')


def test_encoding_embedding(opts):
    data_path = "../data/tsp/tsp50_cluster_validation_seed4321.pkl"
    save_path = "../mydata/tsp/image/"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    opts = parser.parse_args()
    model, _ = load_model(opts.model)
    input_embedding = model.init_embed_with_output(torch.Tensor(data[0]).unsqueeze(0))[0]
    ori_data = data[0]
    ori_dist_list = []
    embedding_list = []
    embedding_similarity_list = []
    eud2_max = 0
    embedding_max = 0
    for i in range(10):
        current_ori_position = ori_data[i]
        current_eud2_dist_list = []
        for j in range(20):
            pointer_position = ori_data[j]
            eud2_dis = math.sqrt((current_ori_position[0] - pointer_position[0])**2 + (current_ori_position[1] -
                                                                                       pointer_position[1])**2)

            if eud2_dis > eud2_max:
                eud2_max = eud2_dis
            current_eud2_dist_list.append(eud2_dis)
        current_embedding = input_embedding[i]
        current_embedding_list = []
        current_similarity_list = []
        for j in range(20):
            pointer_embedding = input_embedding[j]
            # tensor1 = torch.Tensor(pointer_embedding)
            embedding_dist = torch.Tensor(pointer_embedding).dot(torch.Tensor(current_embedding)).squeeze(0).detach().numpy().max()
            num1 = torch.Tensor(pointer_embedding).dot(torch.Tensor(pointer_embedding)).squeeze(0).detach().numpy().max()
            num2 = torch.Tensor(current_embedding).dot(torch.Tensor(current_embedding)).squeeze(
                0).detach().numpy().max()
            sum = math.sqrt(num1) * math.sqrt(num2)
            current_embedding_list.append(embedding_dist)
            current_similarity_list.append(embedding_dist/sum)
            if embedding_dist > embedding_max:
                embedding_max = embedding_dist
        ori_dist_list.append(current_eud2_dist_list)
        embedding_list.append(current_embedding_list)
        embedding_similarity_list.append(current_similarity_list)
    diff_list = []
    reverse_ori_eud2 = []
    for i in range(10):
        diff_current_list = []
        reverse_ori_eud2_current = []
        for j in range(20):
            embedding_list[i][j] = embedding_list[i][j]/embedding_max
            ori_dist_list[i][j] = ori_dist_list[i][j]/eud2_max
            diff_current_list.append(embedding_similarity_list[i][j] + ori_dist_list[i][j] - 1)
            reverse_ori_eud2_current.append(1 - ori_dist_list[i][j])
        diff_list.append(diff_current_list)
        reverse_ori_eud2.append(reverse_ori_eud2_current)
    sim_min =  1
    for list in embedding_similarity_list:
        if(min(list) < sim_min):
            sim_min = min(list)
    standardization_rev_eud2_dist = copy.deepcopy(reverse_ori_eud2)
    for i in range(10):
        for j in range(20):
            standardization_rev_eud2_dist[i][j] = standardization_rev_eud2_dist[i][j] * (1 - sim_min) + sim_min
    print(embedding_similarity_list)
    print()
    print()
    
    print(standardization_rev_eud2_dist)
    standardization_diff_list = []
    for i in range(10):
        standardization_diff_current_list = []
        for j in range(20):
            standardization_diff_current_list.append(embedding_similarity_list[i][j] -
                                                     standardization_rev_eud2_dist[i][j])
        standardization_diff_list.append(standardization_diff_current_list)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    opts = parser.parse_args()
    test_encoding_embedding(opts)




