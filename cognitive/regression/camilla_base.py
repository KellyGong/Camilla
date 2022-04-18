# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from EduCDM import CDM
from utils import get_diamond_class_info


class Camilla_Base_Net(nn.Module):
    def __init__(self, knowledge_n, user_num, item_num, irf_kwargs=None):
        super(Camilla_Base_Net, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, knowledge_n)
        self.a = nn.Embedding(self.item_num, knowledge_n)
        self.b = nn.Embedding(self.item_num, 1)
        nn.init.uniform_(self.a.weight)

    def forward(self, user, item, input_knowledge_point):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        return self.irf(theta, a, b, input_knowledge_point)

    @classmethod
    def irf(cls, theta, a, b, input_knowledge_point):
        return 1.0 / (1 + torch.exp(b - torch.sum(a * theta * input_knowledge_point, dim=-1)))


class Camilla_Base(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, student_n, exer_n, dataset, args=None):
        super(Camilla_Base, self).__init__()
        self.save_path = 'checkpoint/camilla_base.pt'
        if args:
            self.save_path = f'{args.save_path}/camilla_base.pt'
        self.class_info = torch.from_numpy(get_diamond_class_info().values)

        self.ncdm_net = Camilla_Base_Net(self.class_info.shape[-1], student_n, exer_n)

    def get_knowledge_emb(self, item_id):
        return self.class_info[item_id]
        
    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_mae = 1000
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, disable=True):
                batch_count += 1
                user_id, item_id, y = batch_data
                knowledge_emb = self.get_knowledge_emb(item_id).to(device)
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" %
                  (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                mae, rmse = self.eval(test_data, device=device)
                if mae < best_mae:
                    best_mae = mae
                    self.save()
                print("Camilla-Base [Epoch %d] mae: %.6f, rmse: %.6f" % (epoch_i, mae, rmse))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=True):
            user_id, item_id, y = batch_data
            knowledge_emb = self.get_knowledge_emb(item_id).to(device)
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        self.ncdm_net.train()
        return mean_absolute_error(y_true, y_pred), math.sqrt(mean_squared_error(y_true, y_pred))

    @property
    def model_ability(self):
        # sample_attention = torch.sigmoid(self.irt_net.a.weight.data).mean(dim=0).detach()
        # model_multi_ability = torch.sigmoid(self.irt_net.theta.weight.data).detach()
        # model_ability = torch.einsum('ij,j->i', [model_multi_ability, sample_attention]).tolist()
        sample_attention = self.class_info.float().mean(dim=0).detach()
        model_multi_ability = self.ncdm_net.theta.weight.data.cpu().detach()
        model_ability = torch.einsum('ij,j->i', [model_multi_ability, sample_attention]).tolist()
        return model_ability

    def save(self):
        torch.save(self.ncdm_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.ncdm_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
