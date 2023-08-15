# -*- coding: utf-8 -*-

import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random
import pickle

def sizeof_model(model_dict):
    """Returns the size of the model in bytes."""
    return len(pickle.dumps(model_dict))

def sizeof_data(data):
    """Returns the size of the data in bytes."""
    return len(pickle.dumps(data))

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import send_data, process_isolated, get_data, process_centralized
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    total_communication_central = 0

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    
    device = 'cuda' if args.gpu else 'cpu'
    log_id = args.directory + parameter_list
    # Centralized Learning
    local_epoch = 1 

    # Send Local data to Server 
    total_communication_central = send_data(args)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    train, val, test = process_centralized(args, data)
    central_model = LSTM(args).to(device)
    print(central_model)
    central_weights = central_model.state_dict()
    
    local_update = LocalUpdate(args, train, test)

    central_model = LSTM(args).to(device)
    print("\n")

    for global_round in tqdm.tqdm(range(args.epochs), desc="Global Rounds"):
        local_weights, local_loss, _ = local_update.update_weights(central_model, local_epoch, global_round)
        central_model.load_state_dict(local_weights)
        print(f"\nGlobal Round {global_round + 1} - Loss: {local_loss:.4f}")

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    central_model.load_state_dict(central_weights)

    for cell in selected_cells:
        cell_test = test
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, central_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
 
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('[Centralized Learning] File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
    print(f"Total centralized leanring communication overhead: {total_communication_central / (1024 * 1024):.2f} MB")

