import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math
from EduCDM import CDM
from utils import get_cifar100, get_titanic_class_info


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
        self.save_path = 'checkpoint/camilla_base_cls.pt'
        if args:
            self.save_path = f'{args.save_path}/camilla_base_cls.pt'

        if dataset == 'cifar100_train' or dataset == 'cifar100_test':
            self.get_knowledge_embedding = self.get_cifar_knowledge_embedding
            cifar100 = get_cifar100(dataset)
            self.cifar100_labels = cifar100.targets
            self.ncdm_net = Camilla_Base_Net(max(cifar100.targets) + 1, student_n, exer_n)
        elif dataset == 'titanic':
            self.class_info = torch.from_numpy(get_titanic_class_info().values)
            self.ncdm_net = Camilla_Base_Net(self.class_info.shape[-1], student_n, exer_n)
            self.get_knowledge_embedding = self.get_titanic_knowledge_emb

    def get_cifar_knowledge_embedding(self, item_id):
        knowledge_emb = torch.zeros((len(item_id), max(self.cifar100_labels) + 1))
        for i in range(len(item_id)):
            knowledge_emb[i][self.cifar100_labels[item_id[i]]] = 1.0
        return knowledge_emb
    
    def get_titanic_knowledge_emb(self, item_id):
        return self.class_info[item_id]
    
    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.005, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0.0
        best_loss = 1000.0
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, disable=True):
                batch_count += 1
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb = self.get_knowledge_embedding(item_id).to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            epoch_loss = float(np.mean(epoch_losses))

            print("[Epoch %d] average loss: %.6f" %
                  (epoch_i, epoch_loss))

            if test_data is not None:
                auc, accuracy, f1, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    self.save()
                print("Camilla-Base [Epoch %d] auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, f1, rmse))
            
            else:
                auc, accuracy, f1, rmse = self.eval(train_data, device=device)
                print("Camilla-Base [Epoch %d] train auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, f1, rmse))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save()
                

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=True):
            user_id, item_id, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb = self.get_knowledge_embedding(item_id).to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        self.ncdm_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               f1_score(y_true, np.array(y_pred) >= 0.5, average='macro'), math.sqrt(mean_squared_error(y_true, y_pred))
    
    @property
    def model_ability(self):
        model_ability = torch.sigmoid(self.ncdm_net.theta.weight.data).detach().mean(dim=-1).tolist()
        return model_ability

    def save(self):
        torch.save(self.ncdm_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.ncdm_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
