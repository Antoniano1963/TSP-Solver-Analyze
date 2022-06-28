
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import ctypes
import math


CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

# def get_dist(n1, n2):
# 	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
# 	if isinstance(n1, torch.Tensor):
# 		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
# 	elif isinstance(n1, (list, np.ndarray)):
# 		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
# 	else:
# 		raise TypeError


def generate_data(device, n_samples = 10, n_customer = 20, seed = None):
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
		x[0] -- depot_xy: (batch, 2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
        x[3] -- dist
	"""
	graph = np.random.rand(n_samples, n_customer + 1, 2)
	dist = np.zeros((n_samples, n_customer + 1, n_customer + 1))
	dll = ctypes.cdll.LoadLibrary('./Floyd.so')
	graph_c = graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
	dist_c = dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
	dll.FloydAlgorithm(graph_c, n_samples, n_customer + 1, dist_c)
	# for i in range(n_samples):
	# 	for j in range(n_customer + 1):
	# 		for k in range(n_customer + 1):
	# 			dist[i][j][k] = get_dist(graph[i][j], graph[i][k])
	CAPACITIES = {
		10: 20.,
		20: 30.,
		50: 40.,
		100: 50.
	}
	depot_xy = graph[0:n_samples, 0, :]
	customer_xy = graph[0:n_samples, 1:n_customer+1, :]
	# demand = demand[0:n_samples:, 1:21]
	dist = dist[0:n_samples]
	return (torch.tensor(np.expand_dims(np.array(depot_xy), axis=0), dtype=torch.float).squeeze(0),
		torch.tensor(np.expand_dims(np.array(customer_xy), axis=0), dtype=torch.float).squeeze(0),
		(torch.FloatTensor(n_samples, n_customer).uniform_(0, 9).int() + 1).float() / CAPACITIES[n_customer],
		torch.tensor(np.expand_dims(np.array(dist), axis=0), dtype=torch.float).squeeze(0))
	


class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	"""
	def __init__(self, device, n_samples = 5120, n_customer = 20, seed = None):
		self.tuple = generate_data(device, n_samples, n_customer)

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx], self.tuple[3][idx])

	def __len__(self):
		return self.tuple[0].size(0)

def data_from_txt(path):
	if not os.path.isfile(path):
		raise FileNotFoundError	
	with open(path, 'r') as f:
		lines = list(map(lambda s: s.strip(), f.readlines()))
		customer_xy, demand = [], []
		ZERO, DEPOT, CUSTO, DEMAND = [False for i in range(4)]
		ZERO = True
		for line in lines:
			if(ZERO):
				if(line == 'NODE_COORD_SECTION'):
					ZERO = False
					DEPOT = True

			elif(DEPOT):
				depot_xy = list(map(lambda k: float(k)/100., line.split()))[1:]# depot_xy.append(list(map(int, line.split()))[1:])
				DEPOT = False
				CUSTO = True
				
			elif(CUSTO):
				if(line == 'DEMAND_SECTION'):
					DEMAND = True
					CUSTO = False
					continue
				customer_xy.append(list(map(lambda k: float(k)/100., line.split()))[1:])
			elif(DEMAND):
				if(line == '1 0'):
					continue
				elif(line == 'DEPOT_SECTION'):
					break
				else:
					demand.append(list(map(lambda k: float(k)/100., line.split()))[1])# demand.append(list(map(int, line.split()))[1])
	
	# print(np.array(depot_xy).shape)
	# print(np.array(customer_xy).shape)
	# print(np.array(demand).shape)

	
	
	return (torch.tensor(np.expand_dims(np.array(depot_xy), axis = 0), dtype = torch.float), 
			torch.tensor(np.expand_dims(np.array(customer_xy), axis = 0), dtype = torch.float), 
			torch.tensor(np.expand_dims(np.array(demand), axis = 0), dtype = torch.float))

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device-->', device)
	
	data = generate_data(device, n_samples = 128, n_customer = 20, seed = 123)
	for i in range(3):
	 	print(data[i].dtype)# torch.float32
	 	print(data[i].size())
	
	
	batch, batch_steps, n_customer = 128, 100000, 20
	dataset = Generator(device, n_samples = batch*batch_steps, n_customer = n_customer)
	data = next(iter(dataset))	
	
	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	print('use datalodaer ...')
	for i, data in enumerate(dataloader):
		for j in range(len(data)):
			print(data[j].dtype)# torch.float32
			print(data[j].size())	
		if i == 0:
			break

	path = '../OpenData/A-n53-k7.txt'
	data = data_from_txt(path)
	print('read file ...')
	data = list(map(lambda x: x.to(device), data))
	for da in data:
		print(data[j].dtype)# torch.float32
		print(da.size())
