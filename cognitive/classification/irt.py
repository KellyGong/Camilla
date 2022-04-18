import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math

class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        # nn.init.constant_(self.c.weight, 0.5)
        nn.init.uniform_(self.c.weight)  # 2
        nn.init.uniform_(self.a.weight)

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        return self.irf(theta, a, b, c)

    @classmethod
    def irf(cls, theta, a, b, c, D=1.702):
        return c.clamp(0, 1) + (1 - c.clamp(0, 1)) / (1 + torch.exp((-D * a.clamp_min(1e-3) * (theta - b))))


class IRT():
    def __init__(self, user_num, item_num, args=None):
        super(IRT, self).__init__()
        self.save_path = 'checkpoint/irt_cls.pt'
        if args:
            self.save_path = f'{args.save_path}/irt_cls.pt'
        self.irt_net = IRTNet(user_num, item_num)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        loss_function = nn.BCELoss().to(device)
        self.irt_net = self.irt_net.to(device)
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        best_auc = 0.0
        best_loss = 1000.0
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
            # auc, accuracy = self.eval(train_data, device=device)
            epoch_loss = float(np.mean(losses))
            print("[Epoch %d] loss: %.6f" % (e, epoch_loss))

            if test_data is not None:
                auc, accuracy, f1, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    self.save()
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (e, auc, accuracy, f1, rmse))
            
            else:
                auc, accuracy, f1, rmse = self.eval(train_data, device=device)
                print("IRT [Epoch %d] train auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (e, auc, accuracy, f1, rmse))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save()

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
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               f1_score(y_true, np.array(y_pred) >= 0.5, average='macro'), math.sqrt(mean_squared_error(y_true, y_pred))
    
    @property
    def model_ability(self):
        return self.irt_net.theta.weight.data.detach().tolist()

    def save(self):
        torch.save(self.irt_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.irt_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)

