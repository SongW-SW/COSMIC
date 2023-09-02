import os
from typing import AsyncIterable 
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch
import pickle
import random
def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm+1e-9)
    return out

def standardize(feat, mask):
    scaler = StandardScaler()
    scaler.fit(feat[mask])
    new_feat = torch.FloatTensor(scaler.transform(feat))
    return new_feat
    
    
def preprocess(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)



def save_object(obj, filename):
    with open(filename, 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as fin:
        obj = pickle.load(fin)
    return obj    


class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()

        #self.scores=self.cal_scores()

    def cal_scores(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)

        return scores


    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)

        print(scores)

        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print ('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else :
            print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}
        if os.path.isfile(path+'_neighbor') and os.stat(path+'_neighbor').st_size != 0:
            print ("Exists neighbor file")
            neighbor = torch.load(path+'_neighbor')
        else :
            print ("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
                
            print ("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path+'_neighbor')
            os.system('rm -r {}'.format(path))
            print ("Finish Writing")
        return neighbor

    
class Subgraph:
    #Class for subgraph extraction
    
    def __init__(self, x, edge_index, path, maxsize=50, n_order=10, args=None,id_by_class=None, train_class=None):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize
        
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])), 
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)
        
        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

        self.id_by_class=id_by_class
        self.train_class=train_class

        self.args=args
        
    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
            
    def adjust_edge(self, idx):
        #Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i
            
        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            #edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        #Generate node features for subgraphs
        return self.x[idx]            
    
    def build(self):
        #Extract subgraphs for all nodes
        if os.path.isfile(self.path+'_subgraph') and os.stat(self.path+'_subgraph').st_size != 0:
            print ("Exists subgraph file")
            self.subgraph = torch.load(self.path+'_subgraph')
            return 
        
        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path+'_subgraph')
        
    def search(self, node_list, sample_labels=None, interp=True):
        from torch_geometric.utils import to_dense_adj, dense_to_sparse
        from torch.distributions.beta import Beta
        #Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0

        x_features=[]
        x=[]


        for node in node_list:
            #batch.append(self.subgraph[node])
            batch.append(Data(self.subgraph[node].x,self.subgraph[node].edge_index, torch.ones(self.subgraph[node].edge_index.shape[1])))

            index.append(size)
            size += self.subgraph[node].x.size(0)
            if interp:
                if self.subgraph[node].x.shape[0]<20:
                    x.append(torch.cat([self.subgraph[node].x,torch.zeros([20-self.subgraph[node].x.shape[0],self.subgraph[node].x.shape[1]])],0))
                else:
                    x.append(self.subgraph[node].x)
                x_features.append(self.subgraph[node].x.mean(0))


        #for k in range(10):
        #    scores=self.ppr.cal_scores(k)
        #    print(scores.to_dense())
        #print(1/0)

        x_temp = []
        batch_temp=[]

        #sample_num=15 if self.args.dataset!='Coauthor-CS' else 5
        sample_num=5
        sampled_class = random.sample(range(len(self.train_class)), sample_num )
        sample_idx = []
        for c in sampled_class:
            sample_idx.extend(random.sample(self.id_by_class[c], 5))
        for node in sample_idx:

            #batch.append(self.subgraph[node])
            batch_temp.append(Data(self.subgraph[node].x,self.subgraph[node].edge_index, torch.ones(self.subgraph[node].edge_index.shape[1])))
            if interp:
                if self.subgraph[node].x.shape[0]<20:
                    x_temp.append(torch.cat([self.subgraph[node].x,torch.zeros([20-self.subgraph[node].x.shape[0],self.subgraph[node].x.shape[1]])],0))
                else:
                    x_temp.append(self.subgraph[node].x)
                x_features.append(self.subgraph[node].x.mean(0))




        #N = self.args.n_way

        if interp:
            x_features = torch.stack(x_features, 0)
            #simi=torch.ones([N*K,N*K])*0.5
            simi=torch.sigmoid(-l2_normalize(x_features).matmul(l2_normalize(x_features).t()))

            #batch_temp = Batch().from_data_list(batch)
            batch_temp = Batch().from_data_list(batch_temp)

            dense_adj = to_dense_adj(batch_temp.edge_index, batch_temp.batch,max_num_nodes=20)

            for i in range(len(sample_idx)):

                target_idx = i
                sim = simi[i, target_idx]

                m = Beta(sim*10, 1)
                lambda_value = m.sample(dense_adj[i].shape)

                #lambda_value=0.1

                interp_adj = dense_adj[i] * lambda_value + dense_adj[target_idx] * (1 - lambda_value)
                edge_index, edge_attr = dense_to_sparse(interp_adj)

                lambda_value = m.sample([dense_adj[i].shape[0], 1])

                #lambda_value=0.1
                try:
                    x_interp = x[i] * lambda_value + x_temp[target_idx] * (1 - lambda_value)
                except:
                    pass
                    #print(dense_adj.shape)
                    #print(lambda_value.shape)

                index.append(size)
                size += x_interp.size(0)
                batch.append(Data(x_interp, edge_index, edge_attr))

            #repeat the label
            sample_labels=sample_labels
            for j in range(sample_num):
                sample_labels.extend([j+5]*5)


        batch = Batch().from_data_list(batch)
        index = torch.tensor(index)
        if sample_labels!=None:
            sample_labels= torch.LongTensor(sample_labels)
            return batch, index, sample_labels
        else:
            return batch, index
    
    
    
    
    
   


