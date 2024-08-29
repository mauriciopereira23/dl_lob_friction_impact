import argparse
import os

# load packages
import pandas as pd
import pickle
import numpy as np
import matplotlib

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch import nn, einsum
import torch.nn.functional as F
from tcn import TemporalConvNet
from torch.utils.data import WeightedRandomSampler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model import DTNN

import random
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

import time


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        required = True,
        help = "(int). Number of the GPU to be used.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs='+',
        required = True,
        help = "(list[str]) List of tickers.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs='+',
        required = True,
        help = "(list[int]) List of horizons to generated predictions for.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs='+',
        required = True,
        help = "(list[int]) List of windows to generated predictions for.",
    )
    
    args = vars(parser.parse_args())
    return args

def get_sampler(label):
    class_counts=[0,0,0]
    for i in range(3):
        class_counts[i]=(label==i).sum()
    print('target train 0/1/2: {}/{}/{}'.format(class_counts[0], class_counts[1], class_counts[2]))
    weights=np.zeros(len(label))
    for i in range(len(label)):
        weights[i]=1./class_counts[int(label[i].detach())]
    sampler=WeightedRandomSampler(weights,3)
    return sampler




def prepare_x(data, NF, sequence_stride):
    # df1 = data[:40, :].T
    df1 = data[::sequence_stride,:NF].copy()
    for i in range(20):
        df1 = np.hstack((df1, (df1[:, 2 * i] * df1[:, 2 * i + 1]).reshape(-1, 1)))
    return np.array(df1)


def get_label(data, n_labels, sequence_stride):
    lob = data[::sequence_stride,-n_labels:]
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y




class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T, NF, n_labels, sequence_stride):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data, NF, sequence_stride)
        y = get_label(data, n_labels, sequence_stride)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k]

        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.squeeze(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, checkpoint_path, device, patience):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    counter = 0
    for it in range(epochs):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, checkpoint_path)
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')
        else:
            counter += 1
            

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch + 1}')

        if counter > patience:
            print("early stopping")
            break

    return train_losses, test_losses

def discretize_returns(responses, alphas):
    return (+1)*(responses>=-alphas) + (+1)*(responses>alphas)

def create_dataset(paths, feature_type, alphas, label):
    data_list = []
    for path in paths:
        with np.load(path) as data:
            features = data[feature_type]
            responses = data[label]
            disc_responses = discretize_returns(responses, alphas)
            data = np.hstack([features, disc_responses])
        data_list.append(data)
    dec = np.vstack(data_list)
    return dec

def get_pred_labels(torch_model, data_loader, device):
    targets_list = []
    predictions_list = []
    
    for inputs, targets in data_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
    
        # Forward pass
        predictions = torch_model(inputs)
    
        targets_list.append(targets.cpu().numpy())
        predictions_list.append(predictions.cpu().detach().numpy())

    return np.concatenate(targets_list), np.concatenate(predictions_list)

def evaluate_model(y_true, y_pred, results_filepath, orderbook_updates, horizon, eval_set="test"):
    print(eval_set)
    classification_report_dict = classification_report(y_true, np.argmax(y_pred,axis=1), digits=4, output_dict=True, zero_division=0)
    confusion_matrix_array = confusion_matrix(y_true, np.argmax(y_pred,axis=1))
    categorical_crossentropy = log_loss(y_true, y_pred, labels=[0,1,2])
    pickle.dump(classification_report_dict, open(results_filepath + "/classification_report_" + eval_set + ".pkl", "wb"))
    pickle.dump(confusion_matrix_array, open(results_filepath + "/confusion_matrix_" + eval_set + ".pkl", "wb"))
    pickle.dump(categorical_crossentropy, open(results_filepath + "/categorical_crossentropy_" + eval_set + ".pkl", "wb"))
    
    print("Prediction horizon:", orderbook_updates[horizon], " orderbook updates")
    print("Categorical crossentropy:", categorical_crossentropy)
    print(classification_report_dict)
    print(confusion_matrix_array)


def main():
    params = read_args()
    device = torch.device(f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu")
    print(device)

    ## Hardcoded params - start ##

    ROOT_DIR = "/home/mp422/data_processing"
    Ws = params["windows"]
    
    model_list = ["DTNN"]
    features_list = ["orderbooks"]
    model_inputs_list = ["orderbooks"]
    levels_list = [10]
    
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 10000]
    n_horizons = len(orderbook_updates)
    
    T = 100
    batch_size = 64
    NF = 40
    LR = 1e-4
    depth=3
    train_roll_window = 10
    
    train_days = 4
    test_days = 1
    
    ephocs = 50
    patience = 10
    
    # label_dicts = [
    #      {"train": "uniform_mid_returns",
    #     "val": "uniform_mid_returns",
    #     "test": "uniform_mid_returns"},
    #      {"train": "uniform_mid_returns",
    #     "val": "uniform_mid_returns",
    #     "test": "pit_mid_returns"},
    #      {"train": "uniform_mid_returns",
    #     "val": "uniform_mid_returns",
    #     "test": "bid_ask_returns"}   
    # ]

    label_dicts = [
        {"train_val_label": "pit_mid_returns",
        "test_labels": ["ask_bid_returns","latency_300_mid_returns"]}
        ]

    ## Hardcoded params - end ##

    for m, model_type in enumerate(model_list):
        print("Processing model type: ", model_type)
        model_filepath = os.path.join(ROOT_DIR, "results", model_type)
        os.makedirs(model_filepath, exist_ok=True)
        
        # set local parameters
        features = features_list[m]
        model_inputs = model_inputs_list[m]
        levels = levels_list[m]

        feature_type = model_inputs[:-1] + "_features"
        if model_inputs == "volumes_L3":
            feature_type = model_inputs[:-4] + "_features"

        for TICKER in params["tickers"]:
            print("Processing: ", TICKER)
        
            TICKER_filepath = os.path.join(model_filepath, TICKER)
            os.makedirs(TICKER_filepath, exist_ok=True)
        
            # Splits the files into train / val / test.
            file_list = os.listdir(os.path.join(ROOT_DIR, "data", TICKER))
            file_list = [f for f in file_list if f.endswith("npz") == True]
            file_list.sort()
            dates = [d.split("_")[-1].split(".")[0] for d in file_list]
            
            window_list = []
            window_size = train_days + test_days
            for i, d in enumerate(dates):
                cohort = dates[i:i+window_size]
                if len(cohort) == window_size:
                    window_list.append(cohort)
    
            for W in Ws:
                print("Processing window: ", W)
            
                # select specific window
                for window, window_dates in enumerate(window_list):
                    if window == W:
                        pass
                    else:
                        continue
                    train_val_dates = window_dates[:train_days]
                    test_dates = window_dates[train_days:]
                
                    # set random seeds
                    random.seed(0)
                    np.random.seed(1)
                    
                    # random train-val split
                    random.shuffle(train_val_dates)
                    val_dates = train_val_dates[:train_days // 2]
                    train_dates = train_val_dates[train_days - train_days // 2:]
                
                    window_filepath = os.path.join(TICKER_filepath, "W" + str(window))
                    os.makedirs(window_filepath, exist_ok=True)
                    pickle.dump([val_dates, train_dates, test_dates], open(os.path.join(window_filepath, "val_train_test_dates.pkl"), "wb"))

    
                    for label_dict in label_dicts:
                        train_val_label = label_dict["train_val_label"]

                        train_returns_data_dir = os.path.join(ROOT_DIR, "data", TICKER, train_val_label)
                        train_returns_file_list = os.listdir(train_returns_data_dir)

                        val_returns_data_dir = os.path.join(ROOT_DIR, "data", TICKER, train_val_label)
                        val_returns_file_list = os.listdir(val_returns_data_dir)
                        
                        returns_files = {
                            "train": [os.path.join(train_returns_data_dir, file) for date in train_dates for file in train_returns_file_list if date in file],
                            "val": [os.path.join(val_returns_data_dir, file) for date in val_dates for file in val_returns_file_list if date in file]
                        }

                        data_dir = os.path.join(ROOT_DIR, "data", TICKER)
                        file_list = os.listdir(data_dir)
                        files = {
                            "val": [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file],
                            "train": [os.path.join(data_dir, file) for date in train_dates for file in file_list if date in file],
                            "test": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file]
                        }

                        # print("getting alphas...")
                        # alphas, distributions = get_alphas(returns_files["train"], orderbook_updates)
                        # pickle.dump(alphas, open(os.path.join(window_filepath, "alphas.pkl"), "wb"))
                        # pickle.dump(distributions, open(os.path.join(window_filepath, "distributions.pkl"), "wb"))
                        # imbalances = distributions.to_numpy()
                        alphas = np.zeros(n_horizons)
    
                        
                        # val_distributions = get_class_distributions(returns_files["val"], alphas, orderbook_updates)
                        # pickle.dump(val_distributions, open(os.path.join(window_filepath, "val_distributions.pkl"), "wb"))
                        
                        for test_label in label_dict["test_labels"]:
            
                            test_returns_data_dir = os.path.join(ROOT_DIR, "data", TICKER, test_label)
                            test_returns_file_list = os.listdir(test_returns_data_dir)
            


                            returns_files["test"] = [os.path.join(test_returns_data_dir, file) for date in test_dates for file in test_returns_file_list if date in file]

                            # test_distributions = get_class_distributions(returns_files["test"], alphas, orderbook_updates)
                            # pickle.dump(test_distributions, open(os.path.join(window_filepath + "test_distributions.pkl"), "wb"))
                        
                            # iterate through horizons
                            for horizon in params["horizons"]:
                                start_time = time.time()
                                print(f"-------------------------- ITERATION START TIME {start_time} --------------------------")
                                results_filepath = os.path.join(window_filepath, "h" + str(orderbook_updates[horizon]), "|".join(["train:" + train_val_label, "test:" + test_label]))
                                print(results_filepath)
                                checkpoint_filepath = os.path.join(results_filepath, "weights")
                                os.makedirs(results_filepath, exist_ok=True)
                    
                                train_data_list = []
                                val_data_list = []
                                test_data_list = []
                    
                                dec_train = create_dataset(paths=files["train"], feature_type=feature_type, alphas=alphas, label=train_val_label)
                                dec_val = create_dataset(paths=files["val"], feature_type=feature_type, alphas=alphas, label=train_val_label)
                                dec_test = create_dataset(paths=files["test"], feature_type=feature_type, alphas=alphas, label=test_label)
                                
                                dataset_train = Dataset(data=dec_train, k=horizon, num_classes=3, T=100, NF=NF, n_labels = n_horizons, sequence_stride=train_roll_window)
                                dataset_val = Dataset(data=dec_val, k=horizon, num_classes=3, T=100, NF=NF, n_labels = n_horizons, sequence_stride=train_roll_window)
                                dataset_test = Dataset(data=dec_test, k=horizon, num_classes=3, T=100, NF=NF, n_labels = n_horizons, sequence_stride=1)
                    
                                
                                train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
                                val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
                                test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
                    
                                # print(dataset_train.x.shape, dataset_train.y.shape)
                    
                                # tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)
                                # for x, y in tmp_loader:
                                #     print(x)
                                #     print(y)
                                #     print(x.shape, y.shape)
                                #     break
                    
                                # print(dataset_train.x.shape)
                                
                                model = DTNN(time_slices=dataset_train.x.shape[1], num_classes=3, dim=dataset_train.x.shape[2],
                                                depth=depth, heads=32, mlp_dim=2 * dataset_train.x.shape[2])
             
                                model.to(device)
                    
                                print("Model device:")
                                print(next(model.parameters()).device)
                                
                                # print(summary(model, [1, 100, 60]))
                    
                                criterion = nn.CrossEntropyLoss()
                                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    
                                train_losses, val_losses = batch_gd(model,
                                                                    criterion,
                                                                    optimizer,
                                                                    train_loader,
                                                                    val_loader,
                                                                    epochs=ephocs,
                                                                    checkpoint_path=checkpoint_filepath,
                                                                    device=device,
                                                                    patience=patience)

                                print('------------------ so far so good')
                    
                                model = torch.load(checkpoint_filepath)
                    
                                # Generates predictions.
                                train_targets, train_predictions = get_pred_labels(torch_model=model, data_loader=train_loader, device=device)
                                val_targets, val_predictions = get_pred_labels(torch_model=model, data_loader=val_loader, device=device)
                                test_targets, test_predictions = get_pred_labels(torch_model=model, data_loader=test_loader, device=device)
                    
                                # Calculates evaluation metrics.
                                evaluate_model(y_true=train_targets, y_pred=train_predictions, results_filepath=results_filepath, orderbook_updates=orderbook_updates, horizon=horizon, eval_set="train")
                                evaluate_model(y_true=val_targets, y_pred=val_predictions, results_filepath=results_filepath, orderbook_updates=orderbook_updates, horizon=horizon, eval_set="val")
                                evaluate_model(y_true=test_targets, y_pred=test_predictions, results_filepath=results_filepath, orderbook_updates=orderbook_updates, horizon=horizon, eval_set="test")

                                end_time = time.time()
                                print(f"-------------------------- ITERATION END TIME {end_time} --------------------------")
                                print(f"-------------------------- TIME ELAPSED {end_time - start_time} seconds --------------------------")

if __name__ == "__main__":
    main()