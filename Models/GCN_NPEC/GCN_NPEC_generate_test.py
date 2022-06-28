import math
import time

from GCN_data import Generator, GeneratorTSP
from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP
from GCN_NPEC import GCN_NPEC
import pickle
import os
import numpy as np
import torch
from tqdm import tqdm

def rollout(model, dataset, batch=32, disable_tqdm=False):
    costs_list = []
    dataloader = DataLoader(dataset, batch_size=batch)
    # model.eval()
    for inputs in tqdm(dataloader, disable=disable_tqdm, desc='Rollout greedy execution validate'):
        with torch.no_grad():
            cost, _, _, _, _ = model(inputs, decode_type='greedy')
            costs_list.append(cost)
    return torch.cat(costs_list, 0)



def test_GCN_NPEC_instp(dateset_size=1000, graph_size=20, random_seed=1234):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model = GCN_NPEC(n_customer=graph_size).to(device_gpu)
    model.load_state_dict(torch.load('./GCN_Epoch/VRP50_train/0523_22_40/train_epoch280.pt'))
    dataset = GeneratorTSP('cuda:0', dateset_size, graph_size)
    # Make var works for dicts
    # Run the model
    model.eval()
    cost = rollout(model, dataset)
    costs = cost.cpu().numpy()
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Min cost: {}".format(np.min(costs)))
    costs_dict = dict()
    costs_dict["mean"] = np.mean(costs)
    costs_dict["variance"] = 2 * np.std(costs) / np.sqrt(len(costs))
    return np.mean(costs)


if __name__ == "__main__":
    device_gpu = torch.device("cuda:0")

    costs_list = []
    begin = 5
    end = 100
    step = 5
    seed = 1234
    for i in range(begin, end+5, step):
        print(i)
        start = time.time()
        costs_list.append(test_GCN_NPEC_instp(500, i, seed))
        print("time")
        print(time.time() - start)
    filename = "GCN_NPEC_{}_{}_{}_{}.pkl".format(begin, end, step, seed)
    filedir = "./GCN_NPEC_TSP_Result/" + filename
    print(costs_list)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(filename, 'wb') as f:
        pickle.dump(costs_list, f, pickle.HIGHEST_PROTOCOL)
    print(costs_list)