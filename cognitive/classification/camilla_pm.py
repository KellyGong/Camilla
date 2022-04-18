import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import math
from EduCDM import CDM
import numpy as np
from utils import get_train_dataloader, get_test_dataloader
import torchvision.models as models
import torchvision.transforms as transforms


def load_sample_pretrained_embedding_cifar100(dataset):
    # model = models.resnet18(pretrained=True).to('cuda:2')
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50').to('cuda:2')
    # model = torch.hub.load('facebookresearch/swav:main', 'resnet50w5').to('cuda')
    TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

    if dataset == 'cifar100_train':
        data_loader = get_train_dataloader(
            TRAIN_MEAN,
            TRAIN_STD,
            dataset='cifar100',
            num_workers=1,
            batch_size=128,
            shuffle=False,
            transform=transform_fn
        )

    elif dataset == 'cifar100_test':
        data_loader = get_test_dataloader(
            TRAIN_MEAN,
            TRAIN_STD,
            dataset='cifar100',
            num_workers=1,
            batch_size=128,
            shuffle=False,
            transform=transform_fn
        )
    else:
        raise NotImplementedError

    output = []
    for _, images, _ in tqdm(data_loader):
        images = images.to('cuda:2')
        outputs = model(images)
        outputs = outputs.detach().cpu()
        output.append(outputs)
    output = torch.cat(output, dim=0)

    return output


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # auto learning Q-matrix
        # self.ques_knowledge = nn.Embedding(self.exer_n, self.knowledge_dim)
        
        self.ques_knowledge = nn.Linear(1000, self.knowledge_dim)

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
            if 'weight' in name and 'emb' not in name and 'difficulty' not in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.normal_(param)

    def forward(self, stu_id, input_exercise, input_sample_embedding):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty)
        
        input_x = input_x * torch.softmax(self.ques_knowledge(input_sample_embedding), dim=-1)
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


class Camilla_PM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, dataset):
        super(Camilla_PM, self).__init__()
        self.save_path = f'checkpoint/camilla_pm_{dataset}.pt'
        self.pretrain_sample_embedding_path = f'checkpoint/camilla_pm_{dataset}_sample_embedding.npy'
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.sample_embedding = load_sample_pretrained_embedding_cifar100(dataset)
        np.save(self.save_path, self.sample_embedding.numpy())

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.device = device
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0.0
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, disable=True):
                batch_count += 1
                user_id, item_id, y = batch_data
                input_sample_embedding = self.sample_embedding[item_id].to(device)
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, input_sample_embedding)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.ncdm_net.apply_clipper()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" %
                  (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy, f1, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    self.save()
                print("Camilla-PM [Epoch %d] auc: %.6f, accuracy: %.6f, f1: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, f1, rmse))


    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=True):
            user_id, item_id, y = batch_data
            input_sample_embedding = self.sample_embedding[item_id].to(device)
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, input_sample_embedding)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        self.ncdm_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               f1_score(y_true, np.array(y_pred) >= 0.5, average='macro'), math.sqrt(mean_squared_error(y_true, y_pred))
    
    @property
    def model_ability(self):
        all_item_attention = torch.softmax(self.ncdm_net.ques_knowledge(self.sample_embedding.to(self.device)), dim=-1).mean(dim=0)
        model_multi_ability = torch.sigmoid(self.ncdm_net.student_emb.weight.data.detach())
        model_ability = torch.einsum("ik,k->i", [model_multi_ability, all_item_attention]).tolist()
        return model_ability

    def save(self):
        torch.save(self.ncdm_net.state_dict(), self.save_path)
        logging.info("save parameters to %s" % self.save_path)

    def load(self):
        self.ncdm_net.load_state_dict(torch.load(self.save_path))
        logging.info("load parameters from %s" % self.save_path)
