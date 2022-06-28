from cmath import pi
import subprocess
import pickle
import time
from tqdm import tqdm
from concorde_py.tsp import TSPSolver
import numpy as np



def change_TSBlib_to_dataset(filepath):
    dimension = 0
    next_data = False
    dataset = []
    dataset.append(list())
    max_number = -1
    with open(filepath, "r") as f:
        data = f.readlines()
        for line in data:
            split_line = line.split()
            if "DIMENSION" in split_line[0]:
                dimension = int(split_line[1])
            if "EOF" in split_line[0]:
                next_data = False
            if next_data:
                dataset[0].append([float(split_line[1]), float(split_line[2])])
                max_number = max(max_number, float(split_line[1]), float(split_line[2]))
            if "NODE_COORD_SECTION" in split_line[0]:
                next_data = True
    print(dataset)
    divide_number = 1
    while(divide_number < max_number):
        divide_number = divide_number * 10
    for i in range(len(dataset[0])):
        sub_list = dataset[0][i]
        dataset[0][i] = [sub_list[0]/divide_number, sub_list[1]/divide_number]
    return dataset[0]


if __name__ == "__main__":

    dataset_path = "/home/lhz_models/problem_set/tsp/tsp50_validation_uniform_seed4321/seed4321_size{}_round{}.tsp"
    start_num = 50
    end_num = 50
    step = 5
    dataset_size = 1
    result_list = []
    Total_Running_Time = 0
    neighbour_sum = [0 for i in range(100)]
    for i in range(start_num, end_num + step, step):
        for j in tqdm(range(dataset_size)):
            current_dataset_path = dataset_path.format(i, j)
            solver = TSPSolver.from_tspfile(current_dataset_path)
            solution = solver.solve()
            opt_val = solution.optimal_value
            opt_tour = solution.tour
            current_dataset_np = change_TSBlib_to_dataset(dataset_path)
            current_dataset_np = np.array(current_dataset_np)
            current_dataset_np_index = np.argsort(current_dataset_np)
            print(current_dataset_np_index)

