from mutators.mutator_cluster import Cluster
from mutators.mutator_linear import LinearProject
import matplotlib.pyplot as plt
from mutators.mutator import Mutator
import types
from typing import List
from tqdm import tqdm
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
import time



class Build:
    def __init__(self, dataset_size:int, tsp_size:int, iters:int, mutators:List[Mutator]):
        self.dataset_size= dataset_size
        self.tsp_size = tsp_size
        self.iters = iters
        self.mutators = mutators
        self.dataset = np.random.uniform(size=(dataset_size, tsp_size, 2))
        self.processes = int(cpu_count() / 1.1)

    def single_operation(self, begin:int, end:int) -> np.ndarray:
        print(begin)
        # for i in tqdm(range(self.iters)):
        #     idxs = np.random.randint(0, len(self.mutators) - 1, end - begin)
        #     for j in range(end - begin):
        #         current_mutator = self.mutators[idxs[j]]
        #         self.dataset[j + begin] = current_mutator.do_mutation(self.dataset[j + begin])
        # return self.dataset
        for j in range(end - begin):
            idxs = np.random.randint(0, len(self.mutators) - 1, self.iters)
            current_data = self.dataset[j + begin]
            for i in range(self.iters):
                current_mutator = self.mutators[idxs[i]]
                current_data = current_mutator.do_mutation(current_data)
            self.dataset[j + begin] = current_data
        return self.dataset, begin, end

    def build(self) -> np.ndarray:
        each_length = int(self.dataset_size / self.processes)
        if each_length == 0:
            each_length = 1
        current_index = 0
        with Pool(self.processes) as pool:
            multiple_results = []
            while(True):
                if current_index < self.dataset_size - each_length:
                    multiple_results.append(pool.apply_async(self.single_operation,
                                                         (current_index, current_index + each_length)))
                    current_index += each_length
                else:
                    multiple_results.append(pool.apply_async(self.single_operation,
                                                             (current_index, self.dataset_size)))
                    break
            for i in range(len(multiple_results)):
                result, begin, end = multiple_results[i].get()
                self.dataset[begin:end] = result[begin:end]
        return self.dataset


if __name__ == "__main__":
    start = time.time()
    mutators = [Cluster(0.2), Cluster(0.2)]
    dataset = Build(1280000, 50, 50, mutators).build()
    plt.scatter(dataset[0, :, 0], dataset[0, :, 1], marker='*', color='r', alpha=0.8)
    plt.show()
    print(time.time() - start)
    print(dataset.shape)