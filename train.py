import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from utils_mp import Subgraph, preprocess, save_object
from subgcon import SugbCon
from model import Encoder, Scorer, Pool
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from data_split import *
import time
    
def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--dataset', default='CoraFull')
    parser.add_argument('--batch_size', type=int, help='batch size', default=100)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size default 20', default=20)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=1024)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_shot', type=int, help='k shot', default=1)
    parser.add_argument('--m_qry', type=int, help='m query', default=10)
    parser.add_argument('--test_num', type=int, help='test number', default=100)
    parser.add_argument('--patience', type=int, help='epoch patience number', default=10)
    parser.add_argument('--beta', type=float, help='G-supcon temperture number', default=1.)
    parser.add_argument('--unsup', action='store_true', help='degrade to unsupervised contrastive training (SimCLR).')
    return parser



if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print(args)
    test_num = args.test_num
    n_way = args.n_way
    k_shot = args.k_shot
    m_qry = args.m_qry
    patience = args.patience

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Loading data
    data, train_idx, id_by_class, train_class, dev_class, test_class, degree_inv = split(args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setting up the subgraph extractor
    #ppr_path = './subgraph/' + args.dataset
    current_directory = os.getcwd()
    ppr_path=os.path.join(current_directory, 'subgraph')
    os.makedirs(ppr_path, exist_ok=True)
    ppr_path=os.path.join(ppr_path,args.dataset)
    os.makedirs(ppr_path, exist_ok=True)

    
    subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order, args, id_by_class, train_class)
    subgraph.build()
        
    def train(model, optimizer):
        # Model training
        model.train()
        optimizer.zero_grad()

        args.unsup=False

        if not args.unsup:
            # class_batch = 42 if args.dataset == 'CoraFull' else 24
            # class_batch = len(train_class)
            class_batch = n_way
            sampled_class = random.sample(range(len(train_class)), class_batch)
            sample_idx = []
            for c in sampled_class:
                # sample_idx.extend(random.sample(id_by_class[c], args.batch_size // class_batch))
                sample_idx.extend(random.sample(id_by_class[c], k_shot))
            sample_labels = torch.squeeze(data.y)[sample_idx]
            class_selected = list(set(sample_labels.tolist()))
            sample_labels =[class_selected.index(i) for i in sample_labels]

            batch, index, sample_labels = subgraph.search(sample_idx,sample_labels, interp=True)

            z, summary = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
            loss = model.loss(z, summary, sample_labels)
        else:
            sample_idx = random.sample(range(len(train_idx)), args.batch_size)
            batch, index = subgraph.search(sample_idx, interp=True)
            z, summary = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
            loss = model.loss(z, summary, None)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    
    def get_all_node_emb(model, node_list):
        # Obtain central node embs from subgraphs 
        list_size = node_list.size
        z = torch.Tensor(list_size, args.hidden_size).cuda() 
        group_nb = math.ceil(list_size/args.batch_size)


        for i in range(group_nb):
            maxx = min(list_size, (i + 1) * args.batch_size)
            minn = i * args.batch_size 
            batch, index = subgraph.search(node_list[minn:maxx], interp=False)
            node, _ = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
            z[minn:maxx] = node
        return z
    
    
    def test(model, eval_class, output_ari=False):
        # Model testing
        model.eval()
        #sample downstream few-shot tasks
        test_acc_all = []
        purity_all = 0.
        nmi_all = 0.
        ari_all = 0.
        test_acc = 0.
        n_way=args.n_way
        k_shot = args.k_shot


        train_z=model(data.x.cuda(), data.edge_index.cuda())

        np.save('emb_no_contrast.npy',train_z.detach().cpu().numpy())
        np.save('label.npy',data.y)


        for i in range(test_num):
            test_id_support, test_id_query, test_class_selected = \
                test_task_generator(id_by_class, eval_class, n_way, k_shot, m_qry)

            with torch.no_grad():
                train_z = get_all_node_emb(model, test_id_support)
                test_z = get_all_node_emb(model, test_id_query)
        
            train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(data.y)[test_id_support]])
            test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(data.y)[test_id_query]])




    # save_object(test_z, "./CoraFull/z")
            # save_object(test_y, "./CoraFull/y")

            test_acc = model.test(train_z, train_y, test_z, test_y)
            if output_ari:
                nmi, ari = model.clustering_test(test_z, test_y, n_way)
                nmi_all += nmi/test_num
                ari_all += ari/test_num

            test_acc_all.append(test_acc)
        
        m, s = np.mean(test_acc_all), np.std(test_acc_all)
        interval = 1.96 * (s / np.sqrt(len(test_acc_all)))

        #print("="*40)
        #print('test_acc = {}'.format(m))
        #print('test_interval = {}'.format(interval))
        if output_ari:
            return m, s, interval, nmi_all, ari_all
        else:
            return m, s, interval

    
    def train_eval():
        # Setting up the model and optimizer
        model = SugbCon(
        hidden_channels=args.hidden_size, encoder=Encoder(data.num_features, args.hidden_size,encoder_type='GCN'),
        pool=Pool(in_channels=args.hidden_size),
        scorer=Scorer(args.hidden_size),
        beta=args.beta,
        degree_inv=degree_inv).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print('Start training !!!')
        best_test_acc = 0
        stop_cnt = 0
        best_epoch = 0
        time_begin=time.time()
        for epoch in range(10000):
            loss = train(model, optimizer)
            if epoch%100==0:
                print('epoch = {}, loss = {}'.format(epoch, loss))

            # validation
            if epoch % 10 == 0 and epoch != 0:
                test_acc, _, _ = test(model, dev_class)
                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    stop_cnt = 0
                    #torch.save(model.state_dict(), 'model.pkl')
                else:
                    stop_cnt += 1
                if stop_cnt >= patience:
                    print('Time', time.time()-time_begin, 'Epoch: {}'.format(epoch))
                    break

        # final test
        #model.load_state_dict(torch.load('model.pkl'))
        acc, std, interval, nmi, ari = test(model, test_class, output_ari=True)
        print("Current acc mean: " + str(acc))
        print("Current acc std: " + str(std))
        print('nmi: {:.4f} ari: {:.4f}'.format(nmi,ari))
        return acc, std, interval


    acc_mean = []
    acc_std = []
    acc_interval = []
    for __ in range(5):
        m, s, interval = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
        acc_interval.append(interval)
    print("****"*20)
    print("Final acc: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))
    print("Final acc interval: " + str(np.mean(acc_interval)))



