
from abc import ABC

import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import pandas as pd
from torch.autograd import Variable

import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['GCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            # print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't
            # use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()  # adjacency matrix
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)  # E_k+1 = A_hat * E_k
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LGCACF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LGCACF, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.map_table = self.dataset.map_table

        self.num_users = self.dataset.n_users
        self.num_all_items = self.dataset.n_all_item
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['GCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.aspect = self.config['aspect']

        self.embedding_user = nn.ParameterList(nn.Parameter(torch.randn(self.num_users, self.latent_dim))
                                               for i in range(len(self.aspect)))
        self.embedding_item = nn.ParameterList()
        for i in range(len(self.aspect)):
            self.embedding_item.append(nn.Parameter(torch.randn(self.num_all_items[i], self.latent_dim)))

        if self.config['pretrain'] == 0:
            # nn.init.xavier_uniform_(self.embedding_dict[''].weight, gain=0.1)
            # print('use xavier initializer')

            for i in range(len(self.aspect)):
                nn.init.normal_(self.embedding_user[i], std=0.1)
                nn.init.normal_(self.embedding_item[i], std=0.1)
            world.cprint('use NORMAL distribution initializer')
        else:
            # not implemented
            print('use pre-trained data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()  # adjacency matrix
        print(f"multi-lgn is already to go(dropout:{self.config['dropout']}")

    def computer(self):
        """
        propagate methods for LGCACF
        :return:
        """
        users_emb, items_emb = [], []
        for i in range(len(self.aspect)):
            users_emb.append(self.embedding_user[i])
            items_emb.append(self.embedding_item[i])

        # **************************************************************************
        items_emb_map_ego = []
        users_emb_ego = torch.mean(torch.stack(users_emb, dim=1), dim=1)
        for i in range(len(self.aspect)):
            map_list = list(self.map_table[self.aspect[i]].values)
            map_list = torch.tensor(map_list, dtype=torch.long)
            map_list = Variable(map_list)
            items_emb_map_ego.append(self.embedding_item[i][map_list].to(world.device))
        items_emb_ego = torch.mean(torch.stack(items_emb_map_ego, dim=1), dim=1)
        # ****************************************************************************

        all_emb_ego = torch.cat([users_emb_ego, items_emb_ego])
        # all_emb_ego = F.normalize(all_emb_ego, p=2, dim=1)
        emb = [all_emb_ego]
        if self.config['dropout']:
            if self.training:
                print('dropping')
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            temp_all_emb = []
            if self.A_split:
                print("not implemented")
            else:
                for i in range(len(self.aspect)):
                    temp_all_emb.append(torch.sparse.mm(
                        g_dropped[i], torch.cat([users_emb[i], items_emb[i]])))  # E_k+1(a_i) = A_hat(a_i) * E_k(a_i)
            # ****************************************************************************
            for i in range(len(self.aspect)):
                users_emb[i] = temp_all_emb[i][:self.num_users]
                items_emb[i] = temp_all_emb[i][self.num_users:]
            # ying she
            users_emb_next = torch.mean(torch.stack(users_emb, dim=1), dim=1).to(world.device)
            items_emb_map = [items_emb[0]]
            for i in range(1, len(self.aspect)):
                map_list = list(self.map_table[self.aspect[i]].values)
                # print(len(map_list))
                map_list = torch.tensor(map_list, dtype=torch.long)
                map_list = Variable(map_list)
                items_emb_map.append(items_emb[i][map_list].to(world.device))
            items_emb_next = torch.mean(torch.stack(items_emb_map, dim=1), dim=1)
            all_emb_next = torch.cat([users_emb_next, items_emb_next])
            # ****************************************************************************
            # all_emb_next = F.normalize(all_emb_next, p=2, dim=1)    #
            emb.append(all_emb_next)

        light_out = torch.mean(torch.stack(emb, dim=1), dim=1)  # weighted mean
        # light_out = torch.cat(emb, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_all_items[0]])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.T))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = [self.embedding_user[0][users]]
        pos_emb_ego = [self.embedding_item[0][pos_items]]
        neg_emb_ego = [self.embedding_item[0][neg_items]]   # #######

        for i in range(1, len(self.aspect)):
            users_emb_ego.append(self.embedding_user[i][users])

            pos_map = list(self.map_table.loc[self.map_table[self.aspect[0]].isin(pos_items)][self.aspect[i]])
            pos_map = Variable(torch.tensor(pos_map, dtype=torch.long))
            pos_emb_ego.append(self.embedding_item[i][pos_map])
            neg_map = list(self.map_table.loc[self.map_table[self.aspect[0]].isin(neg_items)][self.aspect[i]])
            neg_map = Variable(torch.tensor(neg_map, dtype=torch.long))
            neg_emb_ego.append(self.embedding_item[i][neg_map])
        # for i in range(len(self.aspect)):
        #     users_emb_ego.append(self.embedding_user[i][users])
        #     pos_emb_ego.append(self.embedding_item[i][pos_items])
        #     neg_emb_ego.append(self.embedding_item[i][neg_items])

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        # reg_loss = (1/2) * (userEmb0.norm(2).pow(2) +
        #                     posEmb0.norm(2).pow(2) +
        #                     negEmb0.norm(2).pow(2)) / float(len(users))
        reg_loss = 0
        for i in range(len(self.aspect)):
            reg_loss += (1/2) * (userEmb0[i].norm(2).pow(2) + posEmb0[i].norm(2).pow(2) + negEmb0[i].norm(2).pow(2))
        reg_loss = reg_loss / float(len(users))

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


