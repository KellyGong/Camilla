from scipy.stats.mstats_extras import compare_medians_ms
import torch
from torch.utils.data import TensorDataset, DataLoader
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
from pathlib import Path
from utils import setup_seed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from utils import save_result_to_excel, correlation, balanced_sampling_cifar_data
import time
import random
import json


current_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())
random_last = random.randint(0, 1000)
current_time_random = current_time + str(random_last)


BATCH_SIZE = 256

DEVICE = 'cuda:4'



DEVSET_SIZE = [i for i in range(1, 50)]
DEVSET_SIZE.extend([i for i in range(50, 111, 10)])

# DEVSET_SIZE = [i for i in range(1, 500, 10)]

def load_response_matrix(dataset, N=None, seed=None, class_balanced=False):
    # class_balanced only for sampling in CIFAR training set
    if dataset == 'molecular':
        data_path = 'data/ESOL.csv'
        return load_regression_response_matrix(data_path, N, seed) + tuple(['regression'])

    elif dataset == 'diamond':
        data_path = 'data/diamond.csv'
        return load_regression_response_matrix(data_path, N, seed) + tuple(['regression'])

    elif dataset == 'cifar100_train':
        data_path = 'data/train_cifar100.csv'
        return load_classification_response_matrix(data_path, N, seed, class_balanced) + tuple(['classification'])

    elif dataset ==  'cifar100_test':
        data_path = 'data/test_cifar100.csv'
        return load_classification_response_matrix(data_path, N, seed, class_balanced) + tuple(['classification'])
    
    elif dataset == 'titanic':
        data_path = 'data/titanic.csv'
        return load_classification_response_matrix(data_path, N, seed, class_balanced) + tuple(['classification'])

    else:
        raise NotImplementedError


def load_two_mutex_response_matrix(N, seed=None, two_dataset=False):

    # if dataset == 'cifar100_train':
    #     data_path = 'data/train_cifar100.csv'
    #     return load_classification_response_matrix(data_path, N, seed, class_balanced) + tuple(['classification'])

    # elif dataset ==  'cifar100_test':
    #     data_path = 'data/test_cifar100.csv'
    #     return load_classification_response_matrix(data_path, N, seed, class_balanced) + tuple(['classification'])

    if two_dataset:
        data_path_valid = 'data/train_cifar100.csv'
        _, valid_response_tuple_list, _, _ = load_classification_response_matrix(data_path_valid, N, seed)
        data_path_test = 'data/test_cifar100.csv'
        _, test_response_tuple_list, _, _ = load_classification_response_matrix(data_path_test)
        
    
    else:
        data_path = 'data/train_cifar100.csv'
        df = pd.read_csv(data_path)
        samples_ids = balanced_sampling_cifar_data(N)
        df_valid = df.loc[samples_ids]
        df_test = df.drop(samples_ids)
        _, valid_response_tuple_list, _, _ = extract_df(df_valid)
        _, test_response_tuple_list, _, _ = extract_df(df_test)
    
    return valid_response_tuple_list, test_response_tuple_list


def load_dataset_type(dataset):
    if dataset == 'molecular':
        return 'regression'

    elif dataset == 'diamond':
        return 'regression'

    elif dataset == 'cifar100_train':
        return 'classification'

    elif dataset ==  'cifar100_test':
        return 'classification'
    
    elif dataset == 'titanic':
        return 'classification'

    else:
        raise NotImplementedError


def load_regression_response_matrix(path, N=None, seed=None):
    df = pd.read_csv(path)

    if N:
        df = df.sample(n=N, random_state=seed, replace=False, axis=0)

    models_name = [df.columns[i] for i in range(1, len(df.columns) - 1)]
    models_mae = []
    models_rmse = []
    
    print(F'--------MODEL NUM: {len(models_name)}------------')
    print(f'--------Sample NUM: {df.shape[0]}------------')
    data = []
    label_value = df[df.columns[-1]].values

    for model in models_name:
        model_predict_value = df[model].values
        models_mae.append(mean_absolute_error(model_predict_value, label_value))
        models_rmse.append(math.sqrt(mean_squared_error(model_predict_value, label_value)))
        relative_error = list(abs(model_predict_value - label_value))
        data.append(relative_error)

    data = np.array(data)

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    response_matrix = 1- data

    rows, cols = np.where(response_matrix >= 0)
    responses = response_matrix[rows, cols]
    response_tuple = [(row, col, response) for (row, col, response) in zip(rows, cols, responses)]
    return response_matrix, response_tuple, models_name, models_mae


def load_classification_response_matrix(path, N=None, seed=None, class_balanced=True):
    df = pd.read_csv(path)

    if N:
        if class_balanced:
            samples_ids = balanced_sampling_cifar_data(N)
            df = df.loc[samples_ids]
        else:
            df = df.sample(n=N, random_state=seed, replace=False, axis=0)
            raise NotImplementedError

    return extract_df(df)


def extract_df(df):
    models_name = [df.columns[i] for i in range(1, len(df.columns))]
    models_accuracy = []
    
    print(f'--------MODEL NUM: {len(models_name)}------------')
    print(f'--------Sample NUM: {df.shape[0]}------------')
    data = []

    for model in models_name:
        model_response = df[model].values
        models_accuracy.append(sum(model_response) / len(model_response))
        data.append(model_response)

    response_matrix = np.array(data)

    rows, cols = np.where(response_matrix >= 0)
    responses = response_matrix[rows, cols]
    response_tuple = [(row, col, response) for (row, col, response) in zip(rows, cols, responses)]
    return response_matrix, response_tuple, models_name, models_accuracy


def transform(response_tuple_list, batch_size, **kwargs):
    x, y, z = list(zip(*response_tuple_list))
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=1, **kwargs)


def diagnose(response_tuple_list, Model_Object, response_type, dataset, args, train_size=None):
    shuffle(response_tuple_list)

    if train_size:
        train_tuple, valid_tuple = train_test_split(response_tuple_list, train_size=train_size)
        valid_tuple, test_tuple = train_test_split(valid_tuple, test_size=0.5)

    
        train_set, valid_set, test_set = transform(train_tuple, BATCH_SIZE), \
                                        transform(valid_tuple, BATCH_SIZE), \
                                        transform(test_tuple, BATCH_SIZE)

    else:
        # train_set = transform(response_tuple_list, BATCH_SIZE)
        # valid_set = None
        # test_set = None
        train_tuple, valid_tuple = train_test_split(response_tuple_list, train_size=0.8)
        train_set, valid_set = transform(train_tuple, BATCH_SIZE), \
                               transform(valid_tuple, BATCH_SIZE)
        test_set = None
    model, item, response = list(zip(*response_tuple_list))
    model_num = max(model) + 1
    item_num = max(item) + 1
    
    if Model_Object == IRT:
        cdm = Model_Object(model_num, item_num, args)
    elif Model_Object == MIRT:
        cdm = Model_Object(model_num, item_num, args.latent_skill_dim, args)
    elif Model_Object == MCD:
        cdm = Model_Object(model_num, item_num, args.latent_skill_dim, DEVICE, args)
    elif Model_Object == NCDM:
        # need load sample skills
        cdm = Model_Object(item_num, model_num, dataset, args)
    elif Model_Object == Camilla_Base:
        # need load sample skills
        cdm = Model_Object(model_num, item_num, dataset, args)
    elif Model_Object == Camilla:
        cdm = Model_Object(args.latent_skill_dim, item_num, model_num, args)
    elif Model_Object == Random_Baseline and response_type == 'classification':
        cdm = Model_Object(model_num, item_num)
    elif Model_Object == Skill_Random and response_type == 'classification':
        cdm = Model_Object(model_num, item_num, dataset)
    else:
        raise NotImplementedError

    cdm.train(train_set, valid_set, epoch=10, device=DEVICE)
    cdm.load()

    if response_type == 'classification':
        if test_set:
            auc, acc, f1, rmse = cdm.eval(test_set, device=DEVICE)
            return auc, acc, f1, rmse, cdm
        else:
            return 0, 0, 0, 0, cdm
    elif response_type == 'regression':
        if test_set:
            mae, rmse = cdm.eval(test_set, device=DEVICE)
            return mae, rmse, cdm
        else:
            return 1000, 1000, cdm
    else:
        raise NotImplementedError


def multiple_diagnose(model, response_type, args):
    if response_type == 'regression':
        return regression_multiple_diagnose(model, args)
    elif response_type == 'classification':
        return classification_multiple_diagnose(model, args)


def classification_multiple_diagnose(model, args):
    dataset = args.dataset
    response_matrix, response_tuple_list, model_name_list, model_gold_value, response_type = load_response_matrix(dataset)
    aucs, accs, f1s, rmses = [], [], [], []
    pearson_cors, pearson_p_values, spearman_cors, spearman_p_values, kendall_cors, kendall_p_values = [], [], [], [], [], []
    partial_correlation_kendall_cors = [[] for i in range(len(DEVSET_SIZE))]
    # partial_ability_kendall_cors_with_gold_value = [[] for i in range(len(DEVSET_SIZE))]
    seed = args.seed
    setup_seed(seed)
    auc, acc, f1, rmse, cdm = diagnose(response_tuple_list, model, response_type, dataset, args, train_size=0.6)
    aucs.append(auc)
    accs.append(acc)
    f1s.append(f1)
    rmses.append(rmse)
    # _, _, _, _, cdm = diagnose(response_tuple_list, model, response_type, dataset)
    model_ability = cdm.model_ability
    pearson_cor, pearson_p_value, spearman_cor, spearman_p_value, kendall_cor, kendall_p_value = correlation(model_gold_value, model_ability)
    pearson_cors.append(pearson_cor)
    pearson_p_values.append(pearson_p_value)
    spearman_cors.append(spearman_cor)
    spearman_p_values.append(spearman_p_value)
    kendall_cors.append(kendall_cor)
    kendall_p_values.append(kendall_p_value)

    if args.partial_rank:

        # for i in range(len(DEVSET_SIZE)):
        #     _, part_response_tuple_list, _, _, _ = load_response_matrix(dataset, DEVSET_SIZE[i], seed, args.balance)
        #     _, _, _, _, partial_cdm = diagnose(part_response_tuple_list, model, response_type, dataset, args)
        #     partial_model_ability = partial_cdm.model_ability
        #     _, _, _, _, partial_kendall_cor, partial_kendall_p_value = correlation(model_ability, partial_model_ability)
        #     partial_correlation_kendall_cors[i].append(partial_kendall_cor)
        
        for i in range(len(DEVSET_SIZE)):
            valid_response_tuple_list, test_response_tuple_list = load_two_mutex_response_matrix(DEVSET_SIZE[i], seed, False)
            _, _, _, _, partial_valid_cdm = diagnose(valid_response_tuple_list, model, response_type, dataset, args)
            partial_valid_model_ability = partial_valid_cdm.model_ability
            _, _, _, _, partial_test_cdm = diagnose(test_response_tuple_list, model, response_type, dataset, args)
            partial_test_model_ability = partial_test_cdm.model_ability
            _, _, _, _, partial_kendall_cor, partial_kendall_p_value = correlation(partial_valid_model_ability, partial_test_model_ability)
            partial_correlation_kendall_cors[i].append(partial_kendall_cor)
    
    save_rank_correlation_on_top(args, spearman_cors, kendall_cors)
    save_rank_value_with_different_sample_num(args, partial_correlation_kendall_cors)
    
    df = pd.DataFrame()
    save_reliability_evaluation_on_top(args, regression=False, auc=aucs, acc=accs, f1=f1s, rmse=rmses)

    df['auc'] = aucs
    df['acc'] = accs
    df['f1'] = f1s
    df['rmse'] = rmses
    df['pearson_cor'] = pearson_cors
    df['pearson_p_value'] = pearson_p_values
    df['spearman_cor'] = spearman_cors
    df['spearman_p_value'] = spearman_p_values
    df['kendall_cor'] = kendall_cors
    df['kendall_p_values'] = kendall_p_values

    if args.partial_rank:
        for i in range(len(DEVSET_SIZE)):
            # partial ability with ability on whole dataset
            df[f'partial_{DEVSET_SIZE[i]}_ability'] = partial_correlation_kendall_cors[i]
    return df

def save_rank_correlation_on_top(args, spearman_cors, kendall_cors):
    save_path = Path(f'{args.dataset}_rank_correlation.csv')

    if save_path.exists():
        save_df = pd.read_csv(save_path)
    else:
        save_df = pd.DataFrame(columns=['Diagnoser', 'Rank Metric', 'Value'])
    
    new_df = pd.DataFrame()
    rank_metric, rank_values, cdm_name = [], [], []
    rank_values.extend(spearman_cors)
    rank_metric.extend(['Spearman' for _ in spearman_cors])
    cdm_name.extend([args.cdm for _ in spearman_cors])
    rank_values.extend(kendall_cors)
    rank_metric.extend(['Kendall Tau' for _ in kendall_cors])
    cdm_name.extend([args.cdm for _ in kendall_cors])
    new_df['Rank Metric'] = rank_metric
    new_df['Value'] = rank_values
    new_df['Diagnoser'] = cdm_name

    save_df = save_df.append(new_df, ignore_index=True)
    save_df = save_df.to_csv(save_path, index=False)


def save_reliability_evaluation_on_top(args, regression=False, mae=None, rmse=None, auc=None, acc=None, f1=None):
    save_path = Path(f'{args.dataset}_reliability_evaluation.csv')

    if regression:
        new_df = pd.DataFrame(columns=['Diagnoser', 'mae', 'rmse'])
        new_df['mae'] = mae
        new_df['rmse'] = rmse
    else:
        new_df = pd.DataFrame(columns=['Diagnoser', 'auc', 'acc', 'f1', 'rmse'])
        new_df['auc'] = auc
        new_df['acc'] = acc
        new_df['f1'] = f1
        new_df['rmse'] = rmse    
    
    cdm_name = []
    cdm_name.extend([args.cdm for _ in rmse])
    new_df['Diagnoser'] = cdm_name
    

    if save_path.exists():
        save_df = pd.read_csv(save_path)
        save_df = save_df.append(new_df, ignore_index=True)
        save_df.to_csv(save_path, index=False)
    else:
        new_df.to_csv(save_path, index=False)

def save_rank_value_with_different_sample_num(args, partial_correlation_kendall_cors):
    if args.balance:
        save_path = Path(f'{args.dataset}_rank_correlation_sample_num_balance.csv')
    else:
        save_path = Path(f'{args.dataset}_rank_correlation_sample_num_unbalance.csv')

    if save_path.exists():
        save_df = pd.read_csv(save_path)
    else:
        save_df = pd.DataFrame(columns=['Diagnoser', 'Sample Size', 'Value'])
    
    new_df = pd.DataFrame()
    sample_size, rank_values, cdm_name = [], [], []

    for i, partial_correlation_kendall_cor in enumerate(partial_correlation_kendall_cors):
        repeat_size = len(partial_correlation_kendall_cor)
        cdm_name.extend([args.cdm for _ in range(repeat_size)])
        rank_values.extend(partial_correlation_kendall_cor)
        sample_size.extend([DEVSET_SIZE[i] for _ in range(repeat_size)])

    new_df['Diagnoser'] = cdm_name
    new_df['Sample Size'] = sample_size
    new_df['Value'] = rank_values

    save_df = save_df.append(new_df, ignore_index=True)
    save_df = save_df.to_csv(save_path, index=False)


def regression_multiple_diagnose(model, args):
    dataset = args.dataset
    response_matrix, response_tuple_list, model_name_list, model_gold_value, response_type = load_response_matrix(dataset)
    maes, rmses = [], []
    pearson_cors, pearson_p_values, spearman_cors, spearman_p_values, kendall_cors, kendall_p_values = [], [], [], [], [], []
    partial_correlation_kendall_cors = [[] for i in range(len(DEVSET_SIZE))]
    partial_ability_kendall_cors_with_gold_value = [[] for i in range(len(DEVSET_SIZE))]
    seed = args.seed
    setup_seed(seed)
    mae, rmse, cdm = diagnose(response_tuple_list, model, response_type, dataset, args, train_size=0.6)
    maes.append(mae)
    rmses.append(rmse)
    model_ability = cdm.model_ability
    pearson_cor, pearson_p_value, spearman_cor, spearman_p_value, kendall_cor, kendall_p_value = correlation(model_gold_value, model_ability)
    pearson_cors.append(pearson_cor)
    pearson_p_values.append(pearson_p_value)
    spearman_cors.append(spearman_cor)
    spearman_p_values.append(spearman_p_value)
    kendall_cors.append(kendall_cor)
    kendall_p_values.append(kendall_p_value)

    if args.partial_rank:
        _, _, cdm = diagnose(response_tuple_list, model, response_type, dataset, args)

        for i in range(len(DEVSET_SIZE)):
            _, part_response_tuple_list, _, _, _ = load_response_matrix(dataset, DEVSET_SIZE[i], seed)
            _, _, partial_cdm = diagnose(part_response_tuple_list, model, response_type, dataset, args)
            partial_model_ability = partial_cdm.model_ability
            _, _, _, _, partial_kendall_cor, partial_kendall_p_value = correlation(model_ability, partial_model_ability)
            partial_correlation_kendall_cors[i].append(partial_kendall_cor)
            _, _, _, _, partial_gold_kendall_cor, partial_kendall_p_value = correlation(model_gold_value, partial_model_ability)
            partial_ability_kendall_cors_with_gold_value[i].append(partial_gold_kendall_cor)

    save_rank_correlation_on_top(args, spearman_cors, kendall_cors)
    save_reliability_evaluation_on_top(args, regression=True, mae=maes, rmse=rmses)
    df = pd.DataFrame()
    df['mae'] = maes
    df['rmse'] = rmses
    df['pearson_cor'] = pearson_cors
    df['pearson_p_value'] = pearson_p_values
    df['spearman_cor'] = spearman_cors
    df['spearman_p_value'] = spearman_p_values
    df['kendall_cor'] = kendall_cors
    df['kendall_p_values'] = kendall_p_values

    if args.partial_rank:
        for i in range(len(DEVSET_SIZE)):
            # partial ability with ability on whole dataset
            df[f'partial_{DEVSET_SIZE[i]}_ability'] = partial_correlation_kendall_cors[i]
        
        for i in range(len(DEVSET_SIZE)):
            df[f'partial_{DEVSET_SIZE[i]}_gold'] = partial_ability_kendall_cors_with_gold_value[i]
    return df

def save_rank(args, spearman_cors, kendall_cors):
    save_path = Path(f'{args.dataset}_rank_correlation.csv')

    if save_path.exists():
        save_df = pd.read_csv(save_path)
    else:
        save_df = pd.DataFrame(columns=['Diagnoser', 'Rank Metric', 'Value'])
    
    new_df = pd.DataFrame()
    rank_metric, rank_values, cdm_name = [], [], []
    rank_values.extend(spearman_cors)
    rank_metric.extend(['Spearman' for _ in spearman_cors])
    cdm_name.extend([args.cdm for _ in spearman_cors])
    rank_values.extend(kendall_cors)
    rank_metric.extend(['Kendall Tau' for _ in kendall_cors])
    cdm_name.extend([args.cdm for _ in kendall_cors])
    new_df['Rank Metric'] = rank_metric
    new_df['Value'] = rank_values
    new_df['Diagnoser'] = cdm_name

    save_df = save_df.append(new_df, ignore_index=True)
    save_df = save_df.to_csv(save_path, index=False)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cdm', choices=['IRT', 'MIRT', 'MCD', 'Camilla', 'NCDM', 'Random', 'Random_Skill',
                                          'Camilla_Base'], default="NCDM", help='cognitive diagnosis models')
    parser.add_argument('--dataset', choices=['molecular', 'diamond', 'cifar100_train', 'cifar100_test', 'titanic'], default='cifar100_train')
    parser.add_argument('--partial_rank', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=True)
    parser.add_argument('--hidden1_size', type=int, default=128)
    parser.add_argument('--hidden2_size', type=int, default=64)
    parser.add_argument('--latent_skill_dim', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--save_path', type=str, default='tmp/result/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_noise', type=float, default=0)
    args = parser.parse_args()
    return args


def save(args):
    with Path(Path(args.save_path) / 'args.json').open('w', encoding='UTF-8') as f:
        json.dump(args.__dict__, f, indent=4)



if __name__ == '__main__':

    args = parser_args()

    args.save_path = f'./{args.dataset}_{args.cdm}_{current_time_random}'

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True)

    print(args)

    save(args)

    response_type = load_dataset_type(args.dataset)

    if response_type == 'regression':
        from cognitive.regression import *
    elif response_type == 'classification':
        from cognitive.classification import *

    if args.cdm == 'IRT':
        model = IRT
    elif args.cdm == 'MIRT':
        model = MIRT
    elif args.cdm == 'MCD':
        model = MCD
    elif args.cdm == 'Camilla':
        model = Camilla
    elif args.cdm == 'NCDM':
        model = NCDM
    elif args.cdm == 'Camilla_Base':
        model = Camilla_Base
    elif response_type == 'classification':
        if args.cdm == 'Random':
            model = Random_Baseline
        elif args.cdm == 'Random_Skill':
            model = Skill_Random
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    df = multiple_diagnose(model, response_type, args)
    
    print(df.describe())

    save_result_to_excel(df.describe(), f'{args.save_path}/{args.dataset}_{args.cdm}.xlsx')
    
