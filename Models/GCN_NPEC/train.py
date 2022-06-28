from traceback import print_tb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import math
from model import Model
# from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator
from config import Config, load_pkl, train_parser
import os


def rollout(model, dataset, batch=256, disable_tqdm=False):
	costs_list = []
	dataloader = DataLoader(dataset, batch_size=batch)
	# model.eval()
	for inputs in tqdm(dataloader, disable=disable_tqdm, desc='Rollout greedy execution validate'):
		with torch.no_grad():
			cost, _, _, _, _ = model(inputs, decode_type='greedy')
			costs_list.append(cost)
	return torch.cat(costs_list, 0)

def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts.batch)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped




def train(cfg, log_path = None):
	torch.backends.cudnn.benchmark = True
	model = Model(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping, cfg.n_customer)
	
	# model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	model.train()
	# if cfg.use_checkpoint:
	# 	checkpoint = torch.load(cfg.checkpoint)
	# 	model.load_state_dict(checkpoint)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.dump_date, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs, device)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, verbose=True)
	# ExpLR = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(0.96 ** epoch, 3e-6))
	val_dataset = Generator(device, int(cfg.batch*cfg.batch_steps/8), cfg.n_customer)
	t1 = time()
	for epoch in range(cfg.epochs):
		ave_total_loss, ave_seq_loss, avg_cla_loss, avg_cost = 0., 0., 0.,  0.
		dataset = Generator(device, cfg.batch*cfg.batch_steps, cfg.n_customer)
		
		bs = baseline.eval_all(dataset)
		bs = bs.view(-1, cfg.batch) if bs is not None else None# bs: (cfg.batch_steps, cfg.batch) or None
		
		dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
		for t, inputs in enumerate(tqdm(dataloader)):
			
			# loss, L_mean = rein_loss(model, inputs, bs, t, device)
			cost, ll, pi, groud_mat, pre_mat = model(inputs, decode_type = 'sampling')
# 			L, ll, pi, groud_mat, pre_mat = model(inputs, decode_type = 'sampling')
			predict_matrix = pre_mat.view(-1, 2).cuda()
			solution_matrix = groud_mat.view(-1).long().cuda()
			crossEntropy = nn.CrossEntropyLoss()
			classification_loss = crossEntropy(predict_matrix, solution_matrix)
			
			b = bs[t] if bs is not None else baseline.eval(inputs, cost)
			se_loss, L_mean = ((cost - b.to(device)) * ll).mean(), cost.mean()

			loss = se_loss + classification_loss
			
			optimizer.zero_grad()
			loss.backward()
			# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
			# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
			grad_norms, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, max_norm=1.0)
			optimizer.step()
			
			ave_total_loss += loss.mean().item()
			ave_seq_loss += se_loss.item()
			avg_cost += cost.mean().item()
			avg_cla_loss += classification_loss
			
			if t%(cfg.batch_verbose) == 0:
				t2 = time()
				print('Epoch %d (batch = %d): ave_total_loss: %1.3f ave_seq_loss: %1.3f avg_cla_loss: %1.3f, seq_cost: %1.3f, baseline_cost: %1.3f, class_loss: %1.3f, grad_norm: %1.3f clipped: %1.3f cost: %1.3f %dmin%dsec'%(
					epoch, t, ave_total_loss/(t+1), ave_seq_loss/(t+1), avg_cla_loss/(t+1), cost.mean().item(), b.mean().item(), classification_loss, grad_norms[0], grad_norms_clipped[0] , avg_cost/(t+1), (t2-t1)//60, (t2-t1)%60))
				print("current learning rate :", optimizer.state_dict()['param_groups'][0]['lr'])

				if cfg.islogger:
					if log_path is None:
						log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,loss,cost\n')
				t1 = time()
				
		baseline.epoch_callback(model, epoch)
		if not os.path.exists('%s%s/%s' % (cfg.weight_dir, cfg.task, cfg.dump_date)):
			os.makedirs('%s%s/%s' % (cfg.weight_dir, cfg.task, cfg.dump_date))
		avg_reward = validate(model, val_dataset, cfg)
		torch.save(model.state_dict(), '%s%s/%s/train_epoch%s.pt' % (cfg.weight_dir, cfg.task, cfg.dump_date, epoch))
		# print(ExpLR.get_last_lr())
		if(ExpLR.get_last_lr()[0] > 3e-6):
			ExpLR.step()
if __name__ == '__main__':
	cfg = load_pkl(train_parser().path)
	train(cfg)	
