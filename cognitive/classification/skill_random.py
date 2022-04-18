'''
Author: your name
Date: 2022-01-02 16:40:44
LastEditTime: 2022-01-04 16:37:27
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /regression-cd/cognitive/classification/skill_random.py
'''
import logging
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math
from utils import get_cifar100, get_titanic_class_info


class Skill_Random():
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, dataset):
        self.user_num = user_num
        self.item_num = item_num
        if dataset == 'cifar100_train' or dataset == 'cifar100_test':
            cifar100 = get_cifar100(dataset)
            self.cifar100_labels = cifar100.targets
            self.train = self.train_cifar100
            self.eval = self.eval_cifar100
        elif dataset == 'titanic':
            self.class_info = get_titanic_class_info().values
            self.train = self.train_titanic
            self.eval = self.eval_titanic
            self.skill_dim = self.transfer_one_hot_ndarr2int(np.ones(self.class_info.shape[-1]))

    def train_cifar100(self, train_data, *args, **kwargs) -> ...:
        user_all_skill_response = np.zeros((self.user_num, 100))
        user_true_response = np.zeros((self.user_num, 100))
        for batch_data in train_data:
            user_id, item_id, response = batch_data
            item_list = item_id.tolist()
            user_list = user_id.tolist()
            response_list = response.tolist()
            for i, user in enumerate(user_list):
                user_all_skill_response[user][self.cifar100_labels[item_list[i]]] += 1
                user_true_response[user][self.cifar100_labels[item_list[i]]] += response_list[i]

        self.user_true_rate = np.array(user_true_response) / np.array(user_all_skill_response)
    
    @staticmethod
    def transfer_one_hot_ndarr2int(ndarray):
        assert 0 <= ndarray.all() <= 1
        count = 0
        for i, one_hot in enumerate(ndarray):
            if one_hot == 1:
                count += 2 ** i
        return count
    
    def train_titanic(self, train_data, *args, **kwargs) -> ...:
        user_all_skill_response = np.zeros((self.user_num, self.skill_dim))
        user_true_skill_response = np.zeros((self.user_num, self.skill_dim))

        user_response = np.zeros(self.user_num)
        user_true_response = np.zeros(self.user_num)
        for batch_data in train_data:
            user_id, item_id, response = batch_data
            item_list = item_id.tolist()
            user_list = user_id.tolist()
            response_list = response.tolist()
            for i, user in enumerate(user_list):
                skill_dim = self.transfer_one_hot_ndarr2int(self.class_info[item_list[i]])
                user_response[user] += 1
                user_true_response[user] += response_list[i]
                user_all_skill_response[user][skill_dim] += 1
                user_true_skill_response[user][skill_dim] += response_list[i]

        self.user_whole_rate = np.array(user_true_response) / np.array(user_response)

        for i in range(self.user_num):
            for j in range(self.skill_dim):
                if user_all_skill_response[i][j] == 0:
                    user_all_skill_response[i][j] = 1
                    user_true_skill_response[i][j] = self.user_whole_rate[i]
                    
        self.user_true_rate = user_true_skill_response / user_all_skill_response

    def eval_cifar100(self, test_data, device="cpu") -> tuple:
        y_pred = []
        y_true = []
        y_rate = []
        for batch_data in test_data:
            user_id, item_id, response = batch_data
            user_true_rate = torch.tensor(self.user_true_rate[user_id.numpy()])
            user_true_class_rate = []
            item_id = item_id.tolist()
            user_true_rate = user_true_rate.tolist()
            for a_user_true_rate, item in zip(user_true_rate, item_id):
                user_true_class_rate.append(a_user_true_rate[self.cifar100_labels[item]])

            random_pred = torch.rand_like(response)

            pred = torch.gt(torch.tensor(user_true_class_rate), random_pred).int()
            y_rate.extend(user_true_class_rate)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        auc, accuracy, f1, rmse = roc_auc_score(y_true, y_rate), accuracy_score(y_true, np.array(y_pred)), \
               f1_score(y_true, np.array(y_pred), average='macro'), math.sqrt(mean_squared_error(y_true, y_rate))

        print("auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (auc, accuracy, f1, rmse))
        return auc, accuracy, f1, rmse
    
    def eval_titanic(self, test_data, device="cpu") -> tuple:
        y_pred = []
        y_true = []
        y_rate = []
        for batch_data in test_data:
            user_id, item_id, response = batch_data
            user_true_rate = torch.tensor(self.user_true_rate[user_id.numpy()])
            user_true_class_rate = []
            item_id = item_id.tolist()
            user_true_rate = user_true_rate.tolist()
            for a_user_true_rate, item in zip(user_true_rate, item_id):
                user_true_class_rate.append(a_user_true_rate[self.transfer_one_hot_ndarr2int(self.class_info[item])])

            random_pred = torch.rand_like(response)

            pred = torch.gt(torch.tensor(user_true_class_rate), random_pred).int()
            y_rate.extend(user_true_class_rate)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        auc, accuracy, f1, rmse = roc_auc_score(y_true, y_rate), accuracy_score(y_true, np.array(y_pred)), \
               f1_score(y_true, np.array(y_pred), average='macro'), math.sqrt(mean_squared_error(y_true, y_rate))

        print("auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (auc, accuracy, f1, rmse))
        return auc, accuracy, f1, rmse
    
    def load(self):
        pass
