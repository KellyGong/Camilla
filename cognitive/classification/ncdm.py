# coding: utf-8
# 2021/4/1 @ WangFei

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


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()
        
        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty)
        input_x = input_x * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = torch.clamp(w, min=0.).detach()


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, exer_n, student_n, dataset, args=None):
        super(NCDM, self).__init__()
        self.save_path = f'checkpoint/ncdm_{dataset}.pt'
        if args:
            self.save_path = f'{args.save_path}/ncdm_{dataset}.pt'
        if dataset == 'cifar100_train' or dataset == 'cifar100_test':
            self.get_knowledge_embedding = self.get_cifar_knowledge_embedding
            cifar100 = get_cifar100(dataset)
            self.cifar100_labels = cifar100.targets
            self.ncdm_net = Net(max(cifar100.targets) + 1, exer_n, student_n)
        elif dataset == 'titanic':
            self.class_info = torch.from_numpy(get_titanic_class_info().values)
            self.ncdm_net = Net(self.class_info.shape[-1], exer_n, student_n)
            self.get_knowledge_embedding = self.get_titanic_knowledge_emb
        
        if args.label_noise > .0:
            self.label_noise = args.label_noise
        else:
            self.label_noise = None
            
    def get_cifar_knowledge_embedding(self, item_id):
        knowledge_emb = torch.zeros((len(item_id), max(self.cifar100_labels) + 1))
        for i in range(len(item_id)):
            knowledge_emb[i][self.cifar100_labels[item_id[i]]] = 1.0
        return knowledge_emb
    
    def get_titanic_knowledge_emb(self, item_id):
        return self.class_info[item_id]
    
    def shuffle_knowledge_emb(self, knowledge_emb):
        random_sample_num = int(self.label_noise * knowledge_emb.size(0))

        position_matrix = torch.randint(low=0, high=100, size=(random_sample_num, 1))

        random_matrix_fill = torch.zeros((random_sample_num, knowledge_emb.size(1)))

        for i, position in enumerate(position_matrix):
            random_matrix_fill[i, position] = 1 

        knowledge_emb[: random_sample_num] = random_matrix_fill

        return knowledge_emb

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
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

                if self.label_noise:
                    knowledge_emb = self.shuffle_knowledge_emb(knowledge_emb)

                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.ncdm_net.apply_clipper()

                epoch_losses.append(loss.mean().item())

            epoch_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" %
                  (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy, f1, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    self.save()
                print("NCDM [Epoch %d] auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, f1, rmse))

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
            user_id, item_id,  y = batch_data
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
        # sample_attention = torch.sigmoid(self.irt_net.a.weight.data).mean(dim=0).detach()
        # model_multi_ability = torch.sigmoid(self.irt_net.theta.weight.data).detach()
        # model_ability = torch.einsum('ij,j->i', [model_multi_ability, sample_attention]).tolist()

        model_ability = torch.sigmoid(self.ncdm_net.student_emb.weight.data).detach().mean(dim=-1).tolist()
        return model_ability

    def save(self):
        torch.save(self.ncdm_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.ncdm_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
