import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, dim=5, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, dim)
        self.a = nn.Embedding(self.item_num, dim)
        self.b = nn.Embedding(self.item_num, 1)
        nn.init.uniform_(self.a.weight)

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        return self.irf(theta, a, b)

    @classmethod
    def irf(cls, theta, a, b):
        return 1.0 / (1 + torch.exp(b - torch.sum(a * theta, dim=-1)))


class MIRT():
    def __init__(self, user_num, item_num, dim, args=None):
        super(MIRT, self).__init__()
        self.save_path = 'checkpoint/mirt.pt'
        if args:
            self.save_path = f'{args.save_path}/mirt.pt'
        self.irt_net = MIRTNet(user_num, item_num, dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.01) -> ...:
        # loss_function = nn.BCELoss().to(device)
        loss_function = nn.MSELoss().to(device)
        self.irt_net = self.irt_net.to(device)
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
        best_mae = 1000

        for e in range(epoch):
            losses = []
            for batch_data in train_data:
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                # assert float(predicted_response.min()) >= .0
                # assert float(predicted_response.max()) <= 1.0
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())

            if test_data is not None:
                mae, rmse = self.eval(test_data, device=device)
                if mae < best_mae:
                    best_mae = mae
                    self.save()
                print("MIRT [Epoch %d] mae: %.6f, rmse: %.6f" % (e, mae, rmse))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in test_data:
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.irt_net.train()
        return mean_absolute_error(y_true, y_pred), math.sqrt(mean_squared_error(y_true, y_pred))

    @property
    def model_ability(self):
        # sample_attention = torch.sigmoid(self.irt_net.a.weight.data).mean(dim=0).detach()
        # model_multi_ability = torch.sigmoid(self.irt_net.theta.weight.data).detach()
        # model_ability = torch.einsum('ij,j->i', [model_multi_ability, sample_attention]).tolist()

        model_ability = torch.sigmoid(self.irt_net.theta.weight.data).detach().mean(dim=-1).tolist()
        return model_ability

    def save(self):
        torch.save(self.irt_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.irt_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
