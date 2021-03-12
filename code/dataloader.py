
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Movie(BasicDataset):
    def __init__(self, config=world.config, path="../data/movielens"):
        super(Movie, self).__init__()
        # train or test
        cprint(f'loading [{path}]')
        self.aspect = config['aspect']
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.n_all_item = []
        test_file = path + '/test.txt'
        train_file = path + '/train.txt'
        item_info_file = path + '/map-table.csv'
        self.path = path
        train_unique_users, train_a_item, train_user = [], [], []
        for i in range(len(self.aspect)):
            train_a_item.append([])
        test_unique_users, test_item, test_user = [], [], []
        self.traindataSize = 0
        self.testdataSize = 0

        self.map_table = pd.read_csv(item_info_file, sep=',', header=0)
        for i in range(len(self.aspect)):
            aspect_i = set(list(self.map_table[self.aspect[i]].values))
            self.n_all_item.append(len(aspect_i))
        self.n_all_item[1] += 1

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    train_user.extend([uid] * len(items))
                    train_a_item[0].extend(items)
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(train_unique_users)
        self.trainUser = np.array(train_user)
        for i in range(1, len(self.aspect)):
            train_aspect_i = list(self.map_table.iloc[train_a_item[0]][self.aspect[i]].values)
            train_a_item[i].extend(train_aspect_i)
        self.trainAllItem = []
        for i in range(len(self.aspect)):
            self.trainAllItem.append(np.array(train_a_item[i]))
        # self.trainAllItem = np.array(train_a_item)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    self.n_user = max(self.n_user, uid)
                    self.testdataSize += len(items)
        self.testUniqueUsers = np.array(test_unique_users)
        self.testUser = np.array(test_user)
        self.testItem = np.array(test_item)
        self.n_user += 1

        self.Graph = None
        print(f"{self.traindataSize} interactions for training")
        print(f"{self.testdataSize} interactions for testing")
        print(f"{world.dataset} Sparsity: "
              f"{(self.traindataSize + self.testdataSize) / self.n_users / self.n_all_item[0]}")

        # bipartite graph
        self.InteractNet = []
        self.users_D = []
        self.all_items_D = []
        for i in range(len(self.aspect)):
            self.InteractNet.append(
                csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainAllItem[i])))
            )
            self.users_D.append(np.array(self.InteractNet[i].sum(axis=1)).squeeze())
            self.users_D[i][self.users_D[i] == 0.] = 1
            self.all_items_D.append(np.array(self.InteractNet[i].sum(axis=0)).squeeze())
            self.all_items_D[i][self.all_items_D[i] == 0.] = 1

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.n_all_item[0]

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def getmaptable(self):
        return self.map_table

    @property
    def getTrainItems(self):
        return self.trainAllItem[0]

    def _split_A_hat(self, A):
        """
        :param A: 
        :return: 
        """

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        :param X: 
        :return: 
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = []
                for i in range(len(self.aspect)):
                    pre_adj_mat.append(sp.load_npz(
                        self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i]
                    ))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")        
                start = time()
                norm_adj = []
                for i in range(len(self.aspect)):
                    # print(self.m_items)
                    adj_mat = sp.dok_matrix((self.n_users + self.n_all_item[i], self.n_users + self.n_all_item[i]),
                                            dtype=np.float32)
                    adj_mat = adj_mat.tolil()
                    R = self.InteractNet[i].tolil()
                    adj_mat[:self.n_users, self.n_users:] = R
                    adj_mat[self.n_users:, :self.n_users] = R.T
                    adj_mat = adj_mat.todok()
                    # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
    
                    rowsum = np.array(adj_mat.sum(axis=1))
                    # print(np.power(rowsum, -0.5))
                    d_inv = np.power(rowsum, -0.5).flatten()
                    d_inv[np.isinf(d_inv)] = 0.
                    d_mat = sp.diags(d_inv)
    
                    #  D_-1/2 * adj_mat * D_-1/2
                    norm_adj_ = d_mat.dot(adj_mat)
                    norm_adj_ = norm_adj_.dot(d_mat)
                    norm_adj_.tocsr()
                    norm_adj.append(norm_adj_)
                end = time()
                print(f'costing {end - start}s, saved norm_mat...')
                for i in range(len(self.aspect)):
                    sp.save_npz(self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i], norm_adj[i])
            
            self.Graph = []
            if self.split is True:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._split_A_hat(norm_adj[i]))
                print("done split matrix")
            else:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._convert_sp_mat_to_sp_tensor(norm_adj[i]))
                    self.Graph[i] = self.Graph[i].coalesce().to(world.device)
                print("don't split the matrix")
        
        return self.Graph
    
    def __build_test(self):
        """
        :return: 
            dict: {user: [item]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserPosItems(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.InteractNet[0][user].nonzero()[1])
        return pos_items


class TaoBao(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    TaoBao dataset
    """
    def __init__(self, config=world.config, path="../data/TaoBao"):
        super(TaoBao, self).__init__()
        # train of test
        cprint(f"loading [{path}]")
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.aspect = config['aspect']
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.path = path
        self.map_table = pd.read_csv(join(path + '/map-table.csv'), sep=',', header=0)
        self.n_user = 69166
        self.n_all_item = [39406, 1671]
        train_data = pd.read_csv(join(path + '/train.txt'), sep=' ', header=None)
        test_data = pd.read_csv(join(path + '/test.txt'), sep=' ', header=None)
        self.trainData = train_data
        self.testData = test_data
        self.trainUser = np.array(train_data[:][0])     # train user list
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainAllItem = []                          # train item of multi aspect info
        for i in range(len(self.aspect)):
            self.trainAllItem.append(np.array(train_data[:][i+1]))
        self.testUser = np.array(test_data[:][0])       # test user list
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(test_data[:][1])       # test item list
        self.Graph = None

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity: {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # bipartite graph
        self.InteractNet = []
        self.users_D = []
        self.all_items_D = []
        for i in range(len(self.aspect)):
            self.InteractNet.append(
                csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainAllItem[i])))
            )
            self.users_D.append(np.array(self.InteractNet[i].sum(axis=1)).squeeze())
            self.users_D[i][self.users_D[i] == 0.] = 1
            self.all_items_D.append(np.array(self.InteractNet[i].sum(axis=0)).squeeze())
            self.all_items_D[i][self.all_items_D[i] == 0.] = 1

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.n_all_item[0]

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDataSize(self):
        return len(self.testUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def getTrainItems(self):
        return self.trainAllItem[0]

    def _split_A_hat(self, A):
        """
        :param A:
        :return:
        """

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        :param X:
        :return:
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row).long()
        col = torch.tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                norm_adj = []
                for i in range(len(self.aspect)):
                    norm_adj.append(sp.load_npz(
                        self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i]
                    ))
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                start = time()
                norm_adj = []
                for i in range(len(self.aspect)):
                    adj_mat = sp.dok_matrix((self.n_users + self.n_all_item[i], self.n_users + self.n_all_item[i]),
                                            dtype=np.float32)
                    adj_mat = adj_mat.tolil()
                    R = self.InteractNet[i].tolil()
                    adj_mat[:self.n_users, self.n_users:] = R
                    adj_mat[self.n_users:, :self.n_users] = R.T
                    adj_mat = adj_mat.todok()
                    # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])      # with/without self-connection

                    rowsum = np.array(adj_mat.sum(axis=1))
                    # print(np.power(rowsum, -0.5))
                    d_inv = np.power(rowsum, -0.5).flatten()
                    d_inv[np.isinf(d_inv)] = 0.
                    d_mat = sp.diags(d_inv)

                    #  D_-1/2 * adj_mat * D_-1/2
                    norm_adj_ = d_mat.dot(adj_mat)
                    norm_adj_ = norm_adj_.dot(d_mat)
                    norm_adj_.tocsr()
                    norm_adj.append(norm_adj_)
                end = time()
                print(f'costing {end - start}s, saved norm_mat...')
                for i in range(len(self.aspect)):
                    sp.save_npz(self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i], norm_adj[i])

            self.Graph = []
            if self.split is True:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._split_A_hat(norm_adj[i]))
                print("done split matrix")
            else:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._convert_sp_mat_to_sp_tensor(norm_adj[i]))
                    self.Graph[i] = self.Graph[i].coalesce().to(world.device)
                print("don't split the matrix")

        return self.Graph

    def __build_test(self):
        """
        :return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        pos_item = []
        for user in users:
            pos_item.append(self.InteractNet[0][user].nonzero()[1])
        return pos_item


class Amazon(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    Amazon dataset
    """
    def __init__(self, config=world.config, path="../data/TaoBao"):
        super(Amazon, self).__init__()
        # train of test
        cprint(f"loading [{path}]")
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.aspect = config['aspect']
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.path = path
        self.map_table = pd.read_csv(join(path + '/map-table.csv'), sep=',', header=0)
        self.n_user = 13201
        self.n_all_item = [14094, 2771, 15]
        train_data = pd.read_csv(join(path + '/train.txt'), sep=' ', header=None)
        test_data = pd.read_csv(join(path + '/test.txt'), sep=' ', header=None)
        self.trainData = train_data
        self.testData = test_data
        self.trainUser = np.array(train_data[:][0])     # train user list
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainAllItem = []                          # train item of multi aspect info
        for i in range(len(self.aspect)):
            self.trainAllItem.append(np.array(train_data[:][i+1]))
        self.testUser = np.array(test_data[:][0])       # test user list
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(test_data[:][1])       # test item list
        self.Graph = None

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity: {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # bipartite graph
        self.InteractNet = []
        self.users_D = []
        self.all_items_D = []
        for i in range(len(self.aspect)):
            self.InteractNet.append(
                csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainAllItem[i])))
            )
            self.users_D.append(np.array(self.InteractNet[i].sum(axis=1)).squeeze())
            self.users_D[i][self.users_D[i] == 0.] = 1
            self.all_items_D.append(np.array(self.InteractNet[i].sum(axis=0)).squeeze())
            self.all_items_D[i][self.all_items_D[i] == 0.] = 1

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.n_all_item[0]

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDataSize(self):
        return len(self.testUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def getTrainItems(self):
        return self.trainAllItem[0]

    def _split_A_hat(self, A):
        """
        :param A:
        :return:
        """

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        :param X:
        :return:
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row).long()
        col = torch.tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                norm_adj = []
                for i in range(len(self.aspect)):
                    norm_adj.append(sp.load_npz(
                        self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i]
                    ))
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                start = time()
                norm_adj = []
                for i in range(len(self.aspect)):
                    adj_mat = sp.dok_matrix((self.n_users + self.n_all_item[i], self.n_users + self.n_all_item[i]),
                                            dtype=np.float32)
                    adj_mat = adj_mat.tolil()
                    R = self.InteractNet[i].tolil()
                    adj_mat[:self.n_users, self.n_users:] = R
                    adj_mat[self.n_users:, :self.n_users] = R.T
                    adj_mat = adj_mat.todok()
                    # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])      # with/without self-connection

                    rowsum = np.array(adj_mat.sum(axis=1))
                    # print(np.power(rowsum, -0.5))
                    d_inv = np.power(rowsum, -0.5).flatten()
                    d_inv[np.isinf(d_inv)] = 0.
                    d_mat = sp.diags(d_inv)

                    #  D_-1/2 * adj_mat * D_-1/2
                    norm_adj_ = d_mat.dot(adj_mat)
                    norm_adj_ = norm_adj_.dot(d_mat)
                    norm_adj_.tocsr()
                    norm_adj.append(norm_adj_)
                end = time()
                print(f'costing {end - start}s, saved norm_mat...')
                for i in range(len(self.aspect)):
                    sp.save_npz(self.path + '/s_pre_adj_mat_%s.npz' % self.aspect[i], norm_adj[i])

            self.Graph = []
            if self.split is True:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._split_A_hat(norm_adj[i]))
                print("done split matrix")
            else:
                for i in range(len(self.aspect)):
                    self.Graph.append(self._convert_sp_mat_to_sp_tensor(norm_adj[i]))
                    self.Graph[i] = self.Graph[i].coalesce().to(world.device)
                print("don't split the matrix")

        return self.Graph

    def __build_test(self):
        """
        :return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        pos_item = []
        for user in users:
            pos_item.append(self.InteractNet[0][user].nonzero()[1])
        return pos_item
