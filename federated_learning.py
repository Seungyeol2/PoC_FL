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
import time
import torch.quantization


def sizeof_model(model_dict):
    """Returns the size of the model in bytes."""
    return len(pickle.dumps(model_dict))

def sizeof_data(data):
    """Returns the size of the data in bytes."""
    return len(pickle.dumps(data))

def sparsify_weights(weights, threshold=1e-2):
    sparsified_weights = {}
    for key, weight in weights.items():
        # Set weights smaller than the threshold to 0.
        mask = (weight.abs() > threshold).float()
        sparsified_weights[key] = weight * mask
    return sparsified_weights

def sparse_update(local_weights, global_weights, threshold=0.8):
    updated_weights = {}
    for name, local_weight in local_weights.items():
        global_weight = global_weights[name]
        delta = local_weight - global_weight
        if torch.norm(delta) > threshold:
            updated_weights[name] = delta
    return updated_weights

def quantize_model_weights(model):
    """
    Quantize the model weights.
    """
    model_to_quantize = model.to('cpu')  # Move the model to CPU before quantization
    quantized_model = torch.quantization.quantize_dynamic(model_to_quantize, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model

def dequantize_model_weights(quantized_model):
    # 동적 양자화는 원래의 모델 구조를 유지하므로 별도의 역변환 단계는 필요하지 않습니다.
    # 그러나 원본 모델 구조를 복사한 후 가중치를 복사하는 것이 좋습니다.
    dequantized_model = LSTM(args) # 여기서 args는 전역 변수로 존재해야 합니다.

    # 가중치 복사
    for name, param in quantized_model.named_parameters():
        if name in dequantized_model.state_dict():
            dequantized_model.state_dict()[name].data.copy_(param.data)
    return dequantized_model

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    total_communication_fl = 0

    device = 'cuda' if args.gpu else 'cpu'
    # print(selected_cells)

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    global_model = LSTM(args).to(device)
    global_model.train()
    print(global_model)
    print("\n")

    global_weights = global_model.state_dict()

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    
    for epoch in tqdm.tqdm(range(args.epochs)):
        start_time = time.time()
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]

            bytes_to_server = 0
            bytes_to_local = 0

            local_model = LocalUpdate(args, cell_train, cell_test)
            
            global_weights =  global_model.state_dict()
            sparse_global_weights = sparsify_weights(global_weights)
            #print(f"global model sends its weights to the local models: {sizeof_data(sparse_global_weights)}") 
            global_model.load_state_dict(sparse_global_weights)
            
            # Quantize the global model before sending
            quantized_global_model = quantize_model_weights(global_model)
            quantized_global_weights = quantized_global_model.state_dict()
            #print(f"Quantize the global model before sending: {sizeof_data(quantized_global_weights)}")
            bytes_to_local += sizeof_data(quantized_global_weights)

            # Dequantize the weights after receiving   
            dequantized_model = dequantize_model_weights(quantized_global_model)
            #print(f"Dequantize the weights after receiving : {sizeof_data(dequantized_model.state_dict())}")
            global_model.load_state_dict(dequantized_model.state_dict())
            global_model.to(device)

            w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch, local_epoch=10)
            
            # When the local model sends its updated weights to the global model
            global_weights = global_model.state_dict()
            w_sparse = sparse_update(w, global_weights)
            bytes_to_server += sizeof_model(w_sparse)
            #print(f"local model sends its updated weights to the global model: {sizeof_model(w_sparse)}") 

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            total_communication_fl += bytes_to_local + bytes_to_server

        print(f"Round {epoch} Total communication_overhead: {total_communication_fl} KB")

        loss_hist.append(sum(cell_loss)/len(cell_loss))

        # Update global model
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        #print(f"Round {epoch} took {time.time() - start_time} seconds.")

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('[Federated Learning] File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
    print(f"Total federated learning communication overhead: {total_communication_fl / (1024 * 1024):.2f} MB")
