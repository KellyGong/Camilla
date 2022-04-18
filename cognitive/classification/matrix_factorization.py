'''
Author: your name
Date: 2021-12-28 21:09:40
LastEditTime: 2022-01-20 20:03:50
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /regression-cd/cognitive/classification/matrix_factorization.py
'''
import logging
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math


class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)


class MCD():
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, latent_dim=5, device="cpu", args=None):
        super(MCD, self).__init__()
        self.save_path = 'checkpoint/mcd_cls.pt'
        if args:
            self.save_path = f'{args.save_path}/mcd_cls.pt'
        self.mf_net = MFNet(user_num, item_num, latent_dim).to(device)

    def train(self, train_data, test_data=None, epoch=100, device="cpu", lr=0.001) -> ...:
        loss_function = nn.BCELoss()
        trainer = torch.optim.Adam(self.mf_net.parameters(), lr)
        best_auc = 0.0
        best_loss = 1000.0
        for e in range(epoch):
            losses = []
            for batch_data in train_data:
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.mf_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            # print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            
            epoch_loss = float(np.mean(losses))
            if test_data is not None:
                auc, accuracy, f1, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    self.save()
                print("MCD [Epoch %d] auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (e, auc, accuracy, f1, rmse))
            
            else:
                auc, accuracy, f1, rmse = self.eval(train_data, device=device)
                print("MCD [Epoch %d] train auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (e, auc, accuracy, f1, rmse))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save()

    def eval(self, test_data, device="cpu") -> tuple:
        self.mf_net.eval()
        y_pred = []
        y_true = []
        for batch_data in test_data:
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.mf_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.mf_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               f1_score(y_true, np.array(y_pred) >= 0.5, average='macro'), math.sqrt(mean_squared_error(y_true, y_pred))
    
    @property
    def model_ability(self):
        model_ability = self.mf_net.user_embedding.weight.data.cpu().detach().mean(dim=-1)
        return model_ability
    
    def save(self):
        torch.save(self.mf_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.mf_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
