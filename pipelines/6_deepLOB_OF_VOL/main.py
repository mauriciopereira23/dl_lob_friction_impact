import argparse
import os
from model import deepOB
import tensorflow as tf
import random
import numpy as np
import time


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required = True,
        help = """(str) Directory where the data files will be written. Should also contain a 
        directory named 'input', with orderbook and message files in separated directories for each ticker.""",
    )
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
        help = "(list[str]) List of tickers to generate predictions for.",
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



def main():
    params = read_args()
    visible_gpus = tf.config.experimental.get_visible_devices("GPU")
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    print("This machine has", len(visible_gpus), "visible gpus.")
    print("This machine has", len(physical_gpus), "physical gpus.")
    if visible_gpus:
        try:
            # Use only one GPUs
            tf.config.set_visible_devices(visible_gpus[params["device"]], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    print(f"chosen GPU: {params['device']}")

    ## Hardcoded params - start ##

    ROOT_DIR = params["data_root"]
    Ws = params["windows"]
    
    task = "classification"
    multihorizon = False
    universal=False
    
    T = 100
    
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 10000] 
    n_horizons = len(orderbook_updates)
    
    model_list = ["deepLOB_L2","deepOF_L2","deepVOL_L2"]
    features_list = ["orderbooks", "orderflows","volumes"]
    model_inputs_list = ["orderbooks", "orderflows","volumes"]
    levels_list = [10,10,10]
    
    train_days = 4
    test_days = 1
    
    epochs = 50
    patience = 10
    
    training_verbose = 2
    train_roll_window = 10
    batch_size = 256
    number_of_lstm = 64
    
    decoder = "seq2seq"
    queue_depth = 10
    
    label_dicts = [
        {"train_val_label": "pit_mid_returns",
        "test_labels": ["ask_bid_returns","latency_300_mid_returns"]}
        ]

    ## Hardcoded params - end ##

    # iterate through model types
    for m, model_type in enumerate(model_list):
        model_filepath = os.path.join(ROOT_DIR, "results", model_type)
        os.makedirs(model_filepath, exist_ok=True)
        
        # set local parameters
        features = features_list[m]
        model_inputs = model_inputs_list[m]
        levels = levels_list[m]

        for TICKER in params["tickers"]:
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
                    tf.random.set_seed(2)
                    
                    # random train-val split
                    random.shuffle(train_val_dates)
                    val_dates = train_val_dates[:train_days // 2]
                    train_dates = train_val_dates[train_days - train_days // 2:]
                
                    window_filepath = os.path.join(TICKER_filepath, "W" + str(window))
                    os.makedirs(window_filepath, exist_ok=True)
                    
    
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

                        alphas = np.zeros(n_horizons)
                        imbalances = None
    
                        for test_label in label_dict["test_labels"]:
            
                            test_returns_data_dir = os.path.join(ROOT_DIR, "data", TICKER, test_label)
                            test_returns_file_list = os.listdir(test_returns_data_dir)
            


                            returns_files["test"] = [os.path.join(test_returns_data_dir, file) for date in test_dates for file in test_returns_file_list if date in file]

                        
                            # iterate through horizons
                            for horizon in params["horizons"]:
                                start_time = time.time()
                                print(f"-------------------------- ITERATION START TIME {start_time} --------------------------")
                                results_filepath = os.path.join(window_filepath, "h" + str(orderbook_updates[horizon]), "|".join(["train:" + train_val_label, "test:" + test_label]))
                                checkpoint_filepath = os.path.join(results_filepath, "weights")
                                print(results_filepath)
                                os.makedirs(results_filepath, exist_ok=True)
                    
                                # create model
                                model = deepOB(T = T, #
                                               levels = levels, #
                                               horizon = horizon, #
                                               number_of_lstm = number_of_lstm, #
                                               data_dir = data_dir, #
                                               files = files, #
                                               model_inputs = model_inputs, #
                                               queue_depth = queue_depth,#
                                               task = task, #
                                               alphas = alphas, #
                                               orderbook_updates = orderbook_updates, #
                                               multihorizon = multihorizon, #
                                               decoder = decoder, #
                                               n_horizons = n_horizons, #
                                               train_roll_window = train_roll_window, #
                                               imbalances = imbalances, #
                                               batch_size = batch_size, #
                                               universal = universal, #
                                               train_label = train_val_label,
                                               val_label = train_val_label,
                                               test_label = test_label
                                              )
                    
                                model.create_model()
                    
                                # set random seeds
                                random.seed(0)
                                np.random.seed(1)
                                tf.random.set_seed(2)
                    
                                # train model
                                model.fit_model(epochs = epochs,
                                                checkpoint_filepath = checkpoint_filepath,
                                                verbose = training_verbose,
                                                patience = patience)
                                
                                print("testing model:", results_filepath)
                    
                                # evaluate model
                                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                                     eval_set = "test",
                                                     results_filepath = results_filepath)
                                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                                     eval_set = "train",
                                                     results_filepath = results_filepath)
                                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                                     eval_set = "val",
                                                     results_filepath = results_filepath)
        
                                end_time = time.time()
                                print(f"-------------------------- ITERATION END TIME {end_time} --------------------------")
                                print(f"-------------------------- TIME ELAPSED {end_time - start_time} seconds --------------------------")

if __name__ == "__main__":
    main()