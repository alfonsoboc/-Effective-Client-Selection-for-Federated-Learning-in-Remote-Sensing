#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
from collections import Counter
from torch.distributions.dirichlet import Dirichlet
import torch
import cvxopt
from cvxopt import matrix, solvers
from language_utils import shake_process_x, shake_process_y, sent_process_x, get_word_emb_arr
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset

def mnist_iid(dataset, num_users, rs):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(rs.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, shards_per_client, rs):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards
    num_shards = shards_per_client * num_users
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    return dict_users

def mnist_noniid_unequal(dataset, num_users, rs):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1000, len(dataset) // 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = rs.randint(min_shard, max_shard + 1,
                                   size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(rs.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(rs.choice(idx_shard, shard_size,
                                     replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(rs.choice(idx_shard, shard_size,
                                     replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(rs.choice(idx_shard, shard_size,
                                     replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users

def cifar_iid(dataset, num_users, rs):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) // num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(rs.choice(all_idxs, num_items,
                                      replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users, shards_per_client, rs):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = shards_per_client * num_users
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    return dict_users

def Dirichlet_noniid(dataset, num_users, alpha, rs):
    """
    Sample dataset with dirichlet distribution and concentration parameter alpha
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    num_classes = np.max(labels) + 1
    labels_idxs = []
    prior_class_distribution = np.zeros(num_classes)
    b = np.zeros(num_classes)
    for i in range(num_classes):
        labels_idxs.append(idxs[labels == i])
        prior_class_distribution[i] = len(labels_idxs[i]) / len(dataset)
        b[i] = len(labels_idxs[i])
    
    data_ratio = np.zeros([num_classes, num_users])
    if isinstance(alpha, list):
        for i in range(num_users):
            data_ratio[:, i] = rs.dirichlet(prior_class_distribution * alpha[i])
    else:
        data_ratio = np.transpose(rs.dirichlet(prior_class_distribution * alpha, size=num_users))
    A = matrix(data_ratio)
    b = matrix(b)
    G = matrix(-np.eye(num_users))
    h = matrix(np.zeros([num_users, 1]))
    P = matrix(np.eye(num_users))
    q = matrix(np.zeros([num_users, 1]))
    results = solvers.qp(P, q, G, h, A, b)
    Client_DataSize = np.array(results['x'])
    Data_Division = data_ratio * np.transpose(Client_DataSize)
    rest = []
    for label in range(num_classes):
        for client in range(num_users):
            data_idx = rs.choice(labels_idxs[label], int(Data_Division[label, client]), replace=False)
            dict_users[client] = np.concatenate([dict_users[client], data_idx], 0)
            labels_idxs[label] = list(set(labels_idxs[label]) - set(data_idx))
        rest = rest + labels_idxs[label]

    rest_clients = rs.choice(range(num_users), len(rest), replace=True)
    
    for n, user in enumerate(rest_clients):
        dict_users[user] = np.append(dict_users[user], rest[n])

    for user in range(num_users):
        rs.shuffle(dict_users[user])
    return dict_users, data_ratio

def shakespeare(data_dir, spc, rs):
    trainx = torch.tensor([], dtype=torch.uint8)
    trainy = torch.tensor([], dtype=torch.uint8)
    testx = torch.tensor([], dtype=torch.uint8)
    testy = torch.tensor([], dtype=torch.uint8)
    try:
        trainx = torch.load(data_dir + 'train/xdata.pt')
        trainy = torch.load(data_dir + 'train/ydata.pt')
        user_groups = torch.load(data_dir + 'train/user_groups.pt')
        testx = torch.load(data_dir + 'test/xdata.pt')
        testy = torch.load(data_dir + 'test/ydata.pt')
        
    except: 
        # prepare training set
        user_groups = {}
        start = 0
        with open(data_dir + 'train/data.json', 'r') as inf:
            data = json.load(inf)
        for n, u in enumerate(tqdm(data['users'])):
            temp_x = shake_process_x(data['user_data'][u]['x'])
            temp_y = shake_process_y(data['user_data'][u]['y'])
            trainx = torch.cat((trainx, torch.tensor(temp_x, dtype=torch.uint8)))
            trainy = torch.cat((trainy, torch.tensor(temp_y, dtype=torch.uint8)))
            user_groups[n] = np.arange(start, start + len(temp_x))
            start += len(temp_x)
        torch.save(trainx, data_dir + 'train/xdata.pt')
        torch.save(trainy, data_dir + 'train/ydata.pt')
        torch.save(user_groups, data_dir + 'train/user_groups.pt')

        with open(data_dir + 'test/data.json', 'r') as inf:
            data = json.load(inf)
        for u in tqdm(data['users']):
            temp_x = shake_process_x(data['user_data'][u]['x'])
            temp_y = shake_process_y(data['user_data'][u]['y'])
            testx = torch.cat((testx, torch.tensor(temp_x, dtype=torch.uint8)))
            testy = torch.cat((testy, torch.tensor(temp_y, dtype=torch.uint8)))
        torch.save(testx, data_dir + 'test/xdata.pt')
        torch.save(testy, data_dir + 'test/ydata.pt')
    
    train_dataset = TensorDataset(trainx.long(), trainy.long())
    test_dataset = TensorDataset(testx.long(), testy.long())
    if spc > 1:
        new_user_groups = {}
        remain_role = set(range(len(user_groups.keys())))
        i = 0
        while len(remain_role) >= spc:
            idxs = []
            s = rs.choice(list(remain_role), spc, replace=False)
            remain_role -= set(s)
            for r in s:
                idxs.append(user_groups[r])
            new_user_groups[i] = np.concatenate(idxs, 0)
            i += 1
        user_groups = new_user_groups
    return train_dataset, test_dataset, user_groups

def sent140(data_dir, spc, rs):
    emb_arr, indd, _ = get_word_emb_arr(data_dir + 'embs.json')
    trainx = torch.tensor([], dtype=torch.uint8)
    trainy = torch.tensor([], dtype=torch.uint8)
    testx = torch.tensor([], dtype=torch.uint8)
    testy = torch.tensor([], dtype=torch.uint8)
    try:
        trainx = torch.load(data_dir + 'train/xdata.pt')
        trainy = torch.load(data_dir + 'train/ydata.pt')
        user_groups = torch.load(data_dir + 'train/user_groups.pt')
        testx = torch.load(data_dir + 'test/xdata.pt')
        testy = torch.load(data_dir + 'test/ydata.pt')
        
    except: 
        user_groups = {}
        start = 0
        with open(data_dir + 'train/data.json', 'r') as inf:
            data = json.load(inf)
        for n, u in enumerate(tqdm(data['users'])):
            temp_x = sent_process_x(data['user_data'][u]['x'], emb_arr, indd, 25)
            temp_y = data['user_data'][u]['y']
            trainx = torch.cat((trainx, torch.tensor(temp_x)))
            trainy = torch.cat((trainy, torch.tensor(temp_y, dtype=torch.uint8)))
            user_groups[n] = np.arange(start, start + len(temp_x))
            start += len(temp_x)
        torch.save(trainx, data_dir + 'train/xdata.pt')
        torch.save(trainy, data_dir + 'train/ydata.pt')
        torch.save(user_groups, data_dir + 'train/user_groups.pt')

        with open(data_dir + 'test/data.json', 'r') as inf:
            data = json.load(inf)
        for u in tqdm(data['users']):
            temp_x = sent_process_x(data['user_data'][u]['x'], emb_arr, indd, 25)
            temp_y = data['user_data'][u]['y']
            testx = torch.cat((testx, torch.tensor(temp_x)))
            testy = torch.cat((testy, torch.tensor(temp_y, dtype=torch.uint8)))
        torch.save(testx, data_dir + 'test/xdata.pt')
        torch.save(testy, data_dir + 'test/ydata.pt')
    
    train_dataset = TensorDataset(trainx.float(), trainy.long())
    test_dataset = TensorDataset(testx.float(), testy.long())
    if spc > 1:
        new_user_groups = {}
        remain_role = set(range(len(user_groups.keys())))
        i = 0
        while len(remain_role) >= spc:
            idxs = []
            s = rs.choice(list(remain_role), spc, replace=False)
            remain_role -= set(s)
            for r in s:
                idxs.append(user_groups[r])
            new_user_groups[i] = np.concatenate(idxs, 0)
            i += 1
        user_groups = new_user_groups
    return train_dataset, test_dataset, user_groups

def eurosatiid(datadir, num_users, rs):
    """
    Load EuroSAT data and sample I.I.D. client data for federated learning
    :param datadir: Directory where the data is stored
    :param num_users: Number of users to split the data among
    :param rs: RandomState object for reproducibility
    :return: train_dataset, test_dataset, user_groups
    """
    # Load data
    print("loading datasets")
    all_data = np.load(datadir + 'all_data_train.npy')
    all_labels = np.load(datadir + 'all_data_train_labels.npy')
    print('eurosat loaded')
    # Create train and test datasets
    train_dataset = TensorDataset(torch.tensor(all_data[:20000]), torch.tensor(all_labels[:20000]))
    test_dataset = TensorDataset(torch.tensor(all_data[20000:]), torch.tensor(all_labels[20000:]))
    print("eurosat tensored")
    # Sample I.I.D user data from EuroSAT
    num_items = len(train_dataset) // num_users
    all_idxs = np.arange(len(train_dataset))
    user_groups = {}

    for i in range(num_users):
        print(f'assigning {i}')
        user_groups[i] = set(rs.choice(all_idxs, num_items, replace=False))
        all_idxs = np.setdiff1d(all_idxs, list(user_groups[i]))
    print('distribution of data among clients:\n', {c:len(user_groups[c]) for c in user_groups.keys()})
    unique_valuesd, countsd = np.unique(all_labels[:20000], return_counts=True)
    print('distribution of data among classes:\n', dict(zip(unique_valuesd, countsd)))     
    return train_dataset, test_dataset, user_groups

def eurosatpar(datadir, num_users, shards_per_client, rs):
    """
    Load EuroSAT data and sample non-I.I.D. client data for federated learning
    :param datadir: Directory where the data is stored
    :param num_users: Number of users to split the data among
    :param shards_per_client: Number of shards per client
    :param rs: RandomState object for reproducibility
    :return: train_dataset, test_dataset, user_groups
    """
    # Load data
    all_data = np.load(datadir + 'all_data_train.npy')
    all_labels = np.load(datadir + 'all_data_train_labels.npy')
    
    # Create train and test datasets
    train_dataset = TensorDataset(torch.tensor(all_data[:20000]), torch.tensor(all_labels[:20000]))
    test_dataset = TensorDataset(torch.tensor(all_data[20000:]), torch.tensor(all_labels[20000:]))
    
    # Initialize variables
    num_shards = shards_per_client * num_users
    num_imgs = len(train_dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(train_dataset))
    labels = all_labels[:20000]

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    print('distribution of data among clients:\n', {c:len(dict_users[c]) for c in dict_users.keys()})
    unique_valuesd, countsd = np.unique(all_labels[:20000], return_counts=True)
    print('distribution of data among classes:\n', dict(zip(unique_valuesd, countsd))) 
    client_labels = {}
    for client in range(num_users):
        client_labels[client] = set([all_labels[i] for i in dict_users[client]])
    print('Labels present in each client:\n', client_labels)

    return train_dataset, test_dataset, dict_users

def eurosatnoniid(datadir, num_users):
    """
    Load EuroSAT data and sample heterogeneous client data for federated learning.
    Each client will receive 1400 images of a single class.
    :param datadir: Directory where the data is stored
    :param num_users: Number of users to split the data among (should be <= 10 for 10 classes)
    :return: train_dataset, test_dataset, user_groups
    """
    # Load data
    all_data = np.load(datadir + 'all_data_train.npy')
    all_labels = np.load(datadir + 'all_data_train_labels.npy')

    # Create train and test datasets
    train_dataset = TensorDataset(torch.tensor(all_data[:20000]), torch.tensor(all_labels[:20000]))
    test_dataset = TensorDataset(torch.tensor(all_data[20000:]), torch.tensor(all_labels[20000:]))

    # Initialize user groups
    user_groups = {i: np.array([], dtype=np.int64) for i in range(num_users)}

    # Sort data by class
    for class_label in range(10):  # Assuming there are 10 classes
        class_indices = np.where(all_labels[:20000] == class_label)[0]
        np.random.shuffle(class_indices)  # Shuffle to ensure randomness

        # Assign 1400 images of the current class to each client
        for user_id in range(num_users):
            if user_id < len(class_indices) // 1400:
                start_idx = user_id * 1400
                end_idx = start_idx + 1400
                user_groups[user_id] = np.concatenate(
                    (user_groups[user_id], class_indices[start_idx:end_idx]), axis=0)

    return train_dataset, test_dataset, user_groups

if __name__ == '__main__':
    dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
    num = 100
    d = cifar_noniid(dataset_train, num, 1, np.random.RandomState(1))
    for i in range(num):
        c = Counter(np.array(dataset_train.targets)[d[i]])
        print(c)
        input()
