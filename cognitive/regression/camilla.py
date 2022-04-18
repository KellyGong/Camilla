import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from EduCDM import CDM


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n, args=None):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        if args:
            self.prednet_len1, self.prednet_len2 = args.hidden1_size, args.hidden2_size
        else:
            self.prednet_len1, self.prednet_len2 = 64, 32  # changeable

        super(Net, self).__init__()

        # auto learning Q-matrix
        self.ques_knowledge = nn.Embedding(self.exer_n, self.knowledge_dim)
        
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

    def forward(self, stu_id, input_exercise):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty)
        input_x = input_x * torch.softmax(self.ques_knowledge(input_exercise), dim=-1)
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


class Camilla(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, args=None):
        super(Camilla, self).__init__()
        self.save_path = 'checkpoint/camilla.pt'
        if args:
            self.save_path = f'{args.save_path}/camilla.pt'
        self.ncdm_net = Net(knowledge_n, exer_n, student_n, args)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.01, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_mae = 1000
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, disable=True):
                batch_count += 1
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.ncdm_net.apply_clipper()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" %
                  (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                mae, rmse = self.eval(test_data, device=device)
                if mae < best_mae:
                    best_mae = mae
                    self.save()
                print("Camilla [Epoch %d] mae: %.6f, rmse: %.6f" % (epoch_i, mae, rmse))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=True):
            user_id, item_id, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        self.ncdm_net.train()
        return mean_absolute_error(y_true, y_pred), math.sqrt(mean_squared_error(y_true, y_pred))

    @property
    def model_ability(self):
        all_item_attention = torch.softmax(self.ncdm_net.ques_knowledge.weight.data.detach(), dim=-1).mean(dim=0)
        model_multi_ability = torch.sigmoid(self.ncdm_net.student_emb.weight.data.detach())
        model_ability = torch.einsum("ik,k->i", [model_multi_ability, all_item_attention]).tolist()
        return model_ability

    def save(self):
        torch.save(self.ncdm_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.ncdm_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
