'''
Author: your name
Date: 2022-01-03 14:47:40
LastEditTime: 2022-01-04 16:24:45
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /regression-cd/cognitive/classification/random_baseline.py
'''
import logging
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math


class Random_Baseline():
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num):
        self.user_num = user_num
        self.item_num = item_num
        # self.item_true_rate = np.zeros((item_num))
        self.user_true_rate = np.zeros((user_num))

    def train(self, train_data, *args, **kwargs) -> ...:
        user_all_response = [0] * self.user_num
        user_true_response = [0] * self.user_num

        for batch_data in train_data:
            user_id, item_id, response = batch_data
            item_list = item_id.tolist()
            user_list = user_id.tolist()
            response_list = response.tolist()
            for i, user in enumerate(user_list):
                user_all_response[user] += 1
                user_true_response[user] += response_list[i]

        self.user_true_rate = np.array(user_true_response) / np.array(user_all_response)
    
    def load(self):
        pass

    @property
    def model_ability(self):
        return self.user_true_rate

    def eval(self, test_data, device="cpu") -> tuple:
        y_pred = []
        y_true = []
        y_rate = []
        for batch_data in test_data:
            user_id, item_id, response = batch_data

            # item_true_rate = torch.tensor(self.item_true_rate[item_id.numpy()])
            user_true_rate = torch.tensor(self.user_true_rate[user_id.numpy()])
            random_pred = torch.rand_like(response)

            pred = torch.gt(user_true_rate, random_pred).int()
            y_rate.extend(self.user_true_rate[user_id.numpy()].tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        auc, accuracy, f1, rmse = roc_auc_score(y_true, y_rate), accuracy_score(y_true, np.array(y_pred)), \
               f1_score(y_true, np.array(y_pred), average='macro'), math.sqrt(mean_squared_error(y_true, y_rate))

        print("auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (auc, accuracy, f1, rmse))

        return auc, accuracy, f1, rmse