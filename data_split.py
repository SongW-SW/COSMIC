import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit2, Coauthor, Planetoid, Amazon, DBLP
import random
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
# class_split = {"train": 0.6,"test": 0.4}

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
    'dblp':{"train": 77, 'dev': 30, 'test': 30},
}


class dblp_data():
    def __init__(self):
        self.x=None
        self.edge_index=None
        self.num_nodes=None
        self.y=None
        self.num_edges=None
        self.num_features=7202
class dblp_dataset():
    def __init__(self,data,num_classes):
        self.data=data
        self.num_classes=num_classes

def load_DBLP(root=None, dataset_source='dblp'):
    dataset=dblp_data()
    n1s = []
    n2s = []
    for line in open("./few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        if int(n1)>int(n2):
            n1s.append(int(n1))
            n2s.append(int(n2))



    num_nodes = max(max(n1s), max(n2s)) + 1
    print('nodes num', num_nodes)

    data_train = sio.loadmat("./few_shot_data/{}_train.mat".format(dataset_source))
    data_test = sio.loadmat("./few_shot_data/{}_test.mat".format(dataset_source))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()


    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    dataset.edge_index=torch.tensor([n1s,n2s])
    dataset.y=labels
    dataset.x=features
    dataset.num_nodes=num_nodes
    dataset.num_edges=dataset.edge_index.shape[1]

    return dblp_dataset(dataset, num_classes=80+27+30)

def split(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='~/dataset/' + dataset_name, name="Cora")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='~/dataset/' + dataset_name, name="CiteSeer")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Amazon-Computer':
        dataset = Amazon(root='~/dataset/' + dataset_name, name="Computers")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Coauthor-CS':
        dataset = Coauthor(root='~/dataset/' + dataset_name, name="CS")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Reddit':
        dataset = Reddit2(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name = dataset_name, root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'dblp':
        dataset = load_DBLP(root='./few_shot_data/')
        num_nodes=dataset.data.num_nodes
    else:
        print("Dataset not support!")
        exit(0)
    data = dataset.data
    class_list = [i for i in range(dataset.num_classes)]
    print("********" * 10)





    train_num = class_split[dataset_name]["train"]
    dev_num = class_split[dataset_name]["dev"]
    test_num = class_split[dataset_name]["test"]

    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]
    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)


    train_idx = []
    for cla in train_class:
        train_idx.extend(id_by_class[cla])

    degree_inv = num_nodes / (dataset.data.num_edges * 2)

    return data, np.array(train_idx), id_by_class, train_class, dev_class, test_class, degree_inv


def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



