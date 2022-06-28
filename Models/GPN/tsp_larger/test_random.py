import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref

class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
    
    
    def forward(self, x, h, c):       # query and reference
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c

class GPN(nn.Module):
    
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        
        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        
        
        # =============================
        # vector context
        # =============================
        
        x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (B, size)
        X_all = X_all - x_expand
        
        # the weights share across all the cities
        x = self.embedding_x(x)
        context = self.embedding_all(X_all)
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        
        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.view(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context/(self.city_size-1)))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context/(self.city_size-1)))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context/(self.city_size-1)))
        
        
        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        
        u = 100 * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
    
        return F.softmax(u, dim=1), h, c, latent_u

from tqdm import tqdm_notebook
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

load_root = './model/gpn_tsp500/epoch9.pt'
model = torch.load(load_root).cuda()


result_list = []
for i in range(5,105,5):
    B = 25
    total_tour_len = 0
    n_test = int(1000/B)

    # there are 1000 test data
    size = i
    Z = torch.rand(n_test, B, size, 2).clone()


    for m in tqdm(range(n_test)):
        X = Z[m].cuda()
        
        mask = torch.zeros(B,size).cuda()
        
        R = 0
        solution = []
        reward = 0
        
        Y = X.view(B, size, 2)           # to the same batch size
        x = Y[:,0,:]
        h = None
        c = None
        
        for k in range(size):
            
            output, h, c, _ = model(x=x, X_all=Y, h=h, c=c, mask=mask)
            
            idx = torch.argmax(output, dim=1)
            
            x = Y[[i for i in range(B)], idx.data]
            solution.append(x.cpu().numpy())
            
            mask[[i for i in range(B)], idx.data] += -np.inf
        
        solution.append(solution[0])
        graph = np.array(solution)
        route = [x for x in range(size)] + [0]


        for b in range(B):
            best = route.copy()
            # begin 2-opt
            graph_ = graph[:,b,:].copy()
                
            dmatrix = distance.cdist(graph_, graph_, 'euclidean')
            improved = True
            '''
            while improved:
                improved = False
                
                for i in range(size):
                    for j in range(i+2, size+1):
                        
                        old_dist = dmatrix[best[i],best[i+1]] + dmatrix[best[j], best[j-1]]
                        new_dist = dmatrix[best[j],best[i+1]] + dmatrix[best[i], best[j-1]]
                        
                        # new_dist = 1000
                        if new_dist < old_dist:
                            best[i+1:j] = best[j-1:i:-1]
                            # print(opt_tour)
                            improved = True
            '''
            new_tour_len = 0
            for k in range(size):
                new_tour_len += dmatrix[best[k], best[k+1]]

            
            total_tour_len += new_tour_len
        print("sample:{}, new route length:{}".format(m, new_tour_len))
        
    print('total length:', total_tour_len/1000)
    result_list.append(total_tour_len/1000)
print(result_list)