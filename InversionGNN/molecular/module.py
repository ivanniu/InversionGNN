import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np 
sigmoid = torch.nn.Sigmoid() 
from tqdm import tqdm 

from gnn_layer import GraphConvolution, GraphAttention
from chemutils import smiles2graph, vocabulary 
from epo_lp import EPO_LP

torch.manual_seed(4) 
np.random.seed(1)

# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_out, num_layer,property_num):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features = nfeat, out_features = nhid)
        self.gcs = [GraphConvolution(in_features = nhid, out_features = nhid) for i in range(num_layer)]
        self.gc2 = GraphConvolution(in_features = nhid, out_features = property_num)
        # self.dropout = dropout
        from chemutils import vocabulary 
        self.vocabulary_size = len(vocabulary) 
        self.nfeat = nfeat 
        self.nhid = nhid 
        self.n_out = n_out 
        self.num_layer = num_layer 
        # self.embedding = nn.Embedding(self.vocabulary_size, nfeat)
        self.embedding = nn.Linear(self.vocabulary_size, nfeat)
        self.criteria = torch.nn.BCEWithLogitsLoss() 
        #self.criteria = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.device = device 
        self = self.to(device) 

    def switch_device(self, device):
        self.device = device 
        self = self.to(device)

    def forward(self, node_mat, adj, weight):
        '''
            N: # substructure  &  d: vocabulary size

        Input: 
            node_mat:  
                [N,d]     row sum is 1.
            adj:    
                [N,N]
            weight:
                [N]  

        Output:
            scalar   prediction before sigmoid           [-inf, inf]
        '''
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        #print(x.shape)
        #print(weight.shape)
        logits = torch.sum(x * weight.view(-1,1),dim=0) / torch.sum(weight) 
        #print(logits.shape)
        #exit()
        return logits 
        ## without sigmoid 

    def smiles2embed(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        
        ### forward 
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat) ## bug 
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        return torch.mean(x, 0)

    def smiles2twodim(self, smiles):
        embed = self.smiles2embed(smiles)
          

    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        logits = self.forward(node_mat, adj, weight)
        pred = torch.sigmoid(logits) 
        return pred.item() 


    def update_molecule(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0])
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.item())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 

    def update_molecule_split(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)
            #print(pred_y.shape)
            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.ones_like(pred_y))
            #print(target_y.shape)
            cost = self.criteria(pred_y.view(-1,1), target_y.view(-1,1))
            #print(cost.shape)
            #print(cost)
            #exit()
            #cost = self.criteria(pred_yv[-1], target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.detach().cpu().numpy())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        print('===================================================================================')
        return node_indicator_np2, adjacency_weight_np2  #### np.array 
    
    def update_molecule_split_epo(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np, preference, n_property=2):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
        adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
        _, n_params = getNumParams([node_indicator, adjacency_weight])
        epo_lp = EPO_LP(m=n_property, n=n_params, r=preference)

        pred_lst = []
        for i in range(5000): ### 5k 10k
            #print(i)
            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            #opt_mol = torch.optim.SGD([node_indicator, adjacency_weight], lr=1e-3, momentum=0.)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 


            grads = {}
            losses = []
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)
            for i in range(n_property):
                opt_mol.zero_grad()
                pred_y_split = pred_y[i]
                target_y = Variable(torch.ones_like(pred_y_split))
                task_loss = self.criteria(pred_y_split.view(-1,1), target_y.view(-1,1))
                #print(task_loss)
                losses.append(task_loss.data.numpy())
                task_loss.backward(retain_graph=True)

                grads[i] = []
                for param in [node_indicator, adjacency_weight]:
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                #print(grads[0][1].shape)
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            #print(grads_list[0].shape)
            
            G = torch.stack(grads_list)
            GG = G @ G.T
            losses = np.stack(losses)

            try:
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                alpha = None
            if alpha is None:  
                alpha = preference / preference.sum()
                n_linscalar_adjusts += 1

            alpha = n_property * torch.from_numpy(alpha).to(self.device)
            
            
            losses = []
            opt_mol.zero_grad()
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)
            for i in range(n_property):
                pred_y_split = pred_y[i]
                target_y = Variable(torch.ones_like(pred_y_split))
                task_loss = self.criteria(pred_y_split.view(-1,1), target_y.view(-1,1))
                losses.append(task_loss)
            weighted_loss = losses[0]*alpha[0]
            for i in range(1,len(losses)):
                weighted_loss += losses[i] * alpha[i]

            weighted_loss.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        print('===================================================================================')
        return node_indicator_np2, adjacency_weight_np2  #### np.array 

    def update_molecule_interpret(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0])
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            if i==0:
                node_indicator_grad = node_indicator.grad.detach().numpy()
                adjacency_weight_grad = adjacency_weight.grad.detach().numpy() 
            # print(node_indicator.grad.shape)
            # print(adjacency_weight.grad.shape)

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.item())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2, node_indicator_grad, adjacency_weight_grad  #### np.array 


    def update_molecule_v2(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np, 
                                 leaf_extend_idx_pair, leaf_nonleaf_lst):
        (is_nonleaf_np, is_leaf_np, is_extend_np) = node_mask_np
        is_nonleaf = torch.BoolTensor(is_nonleaf_np).to(self.device)
        is_leaf = torch.BoolTensor(is_leaf_np).to(self.device)
        is_extend = torch.BoolTensor(is_extend_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(6000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)  ### [0,1]
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1)
            ### support shrink 
            node_weight[is_nonleaf] = 1 
            node_weight[is_leaf] = torch.cat([normalized_adjacency_weight[x,y].unsqueeze(0) for x,y in leaf_nonleaf_lst])
            node_weight[is_extend] *= node_weight[is_leaf]

            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0])
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[is_nonleaf_np,:] = node_indicator_np[is_nonleaf_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.item())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 

    def update_molecule_epo_v2(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np, 
                                 leaf_extend_idx_pair, leaf_nonleaf_lst, preference, n_property=2):
        (is_nonleaf_np, is_leaf_np, is_extend_np) = node_mask_np
        is_nonleaf = torch.BoolTensor(is_nonleaf_np).to(self.device)
        is_leaf = torch.BoolTensor(is_leaf_np).to(self.device)
        is_extend = torch.BoolTensor(is_extend_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)
        
        node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
        adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
        _, n_params = getNumParams([node_indicator, adjacency_weight])
        epo_lp = EPO_LP(m=n_property, n=n_params, r=preference)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(6000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))
            #opt_mol = torch.optim.SGD([node_indicator, adjacency_weight], lr=1e-3, momentum=0.)

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)  ### [0,1]
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1)
            ### support shrink 
            node_weight[is_nonleaf] = 1 
            node_weight[is_leaf] = torch.cat([normalized_adjacency_weight[x,y].unsqueeze(0) for x,y in leaf_nonleaf_lst])
            node_weight[is_extend] *= node_weight[is_leaf]

            grads = {}
            losses = []
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)
            for i in range(n_property):
                opt_mol.zero_grad()
                pred_y_split = pred_y[i]
                target_y = Variable(torch.ones_like(pred_y_split))
                task_loss = self.criteria(pred_y_split.view(-1,1), target_y.view(-1,1))
                #print(task_loss)
                losses.append(task_loss.data.numpy())
                task_loss.backward(retain_graph=True)

                grads[i] = []
                for param in [node_indicator, adjacency_weight]:
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                #print(grads[0][1].shape)
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            #print(grads_list[0].shape)
            
            G = torch.stack(grads_list)
            GG = G @ G.T
            losses = np.stack(losses)

            try:
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                alpha = None
            if alpha is None:  
                alpha = preference / preference.sum()
                n_linscalar_adjusts += 1

            alpha = n_property * torch.from_numpy(alpha).to(self.device)
            
            
            losses = []
            opt_mol.zero_grad()
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)
            for i in range(n_property):
                pred_y_split = pred_y[i]
                target_y = Variable(torch.ones_like(pred_y_split))
                task_loss = self.criteria(pred_y_split.view(-1,1), target_y.view(-1,1))
                losses.append(task_loss)
            weighted_loss = losses[0]*alpha[0]
            for i in range(1,len(losses)):
                weighted_loss += losses[i] * alpha[i]

            weighted_loss.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[is_nonleaf_np,:] = node_indicator_np[is_nonleaf_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.detach().cpu().numpy())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 

    def learn(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        #print(pred_y)
        cost = self.criteria(pred_y.view(-1,1), target.view(-1,1))
        #cost = self.criteria(pred_yv[-1], target[-1])
        self.opt.zero_grad() 
        cost.backward() 
        self.opt.step() 
        return cost.data.numpy(), pred_y.data.numpy()

    def valid(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        cost = self.criteria(pred_y.view(-1,1), target.view(-1,1))
        #cost = self.criteria(pred_yv[-1], target[-1])
        return cost.data.numpy(), pred_y.data.numpy()

    
class GCNSum(GCN): 
    def forward(self, node_mat, adj, weight):
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        logits = torch.sum(x * weight.view(-1,1))
        return logits 
        ## without sigmoid 


class GCNRegress(GCN):
    def __init__(self, nfeat, nhid, n_out, num_layer):
        super(GCNRegress, self).__init__(nfeat, nhid, n_out, num_layer)
        self.criteria = torch.nn.MSELoss() 

    def forward(self, node_mat, adj, weight):
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        pred = torch.sum(x * weight.view(-1,1))
        return pred  
        ## without sigmoid     


    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        pred = self.forward(node_mat, adj, weight)
        return pred.item() 




if __name__ == "__main__":

    rawdata_file = "raw_data/zinc.tab"

    with open(rawdata_file) as fin:
        lines = fin.readlines()[1:]

    gnn = GCN(nfeat = 50, nhid = 100, n_out = 1, num_layer = 2)













