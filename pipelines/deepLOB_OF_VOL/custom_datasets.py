import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

def CustomtfDataset(files, NF, horizon, n_horizons, model_inputs, task, alphas, multihorizon, data_transform, batch_size, T, roll_window, shuffle, label, teacher_forcing = False):
    """
    Create custom tf.dataset object to be used by model.
    :param files: files with data, list of str
    :param NF: number of features, int
    :param horizon: prediction horizon, between 0 and tot_horizons, int
    :param n_horizons: number of horizons in multihorizon, int
    :param model_inputs: which input is being used "orderbooks", "orderflows", "volumes" or "volumes_L3", str
    :param task: ML task, "regression" or "classification", str
    :param alphas: alphas for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), (tot_horizons,) array
    :param multihorizon: whether the predictions are multihorizon, if True horizon = slice(0, n_horizons), bool
    :param data_transform: transformation to apply to data, bool
                if "normalize_auto": divide by the largest value to scale between 0 & 1
                if "standardize_rolling_n": use the previous n days mean and std to standardize using data in aggregate stats
    :param T: length of lookback window for features, int
    :param batch_size: batch size for dataset, int
    :param roll_window: length of window to roll forward when extracting features/responses, int
    :param shuffle: whether to shuffle dataset, bool
    :param label: indicates which label should be used for the dataset, dict
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder, bool
    :return: tf_dataset: a tf.dataset object 
    """
    # methods to be used
    def scale_fn(x, y):
        if tf.keras.backend.max(x) == 0:
            x = tf.zeros_like(x, dtype=tf.float32)
        else:
            x = x / tf.keras.backend.max(x)
        return x, y

    def add_decoder_input(x, y):
        if teacher_forcing:
            if task == "classification":
                raise ValueError('teacher forcing not yet implemented.')
            elif task == "regression":
                raise ValueError('teacher forcing not yet implemented.')
            else:
                raise ValueError('task must be either classification or regression.')

        if not teacher_forcing:
            if task == "classification":
                # this sets the initial hidden state of the decoder to be y_0 = [0, 0, 0] for classification
                decoder_input_data = tf.zeros_like(y[0:1, :])
            elif task == "regression":
                # this sets the initial hidden state of the decoder to be y_0 = 0 for regression
                decoder_input_data = tf.zeros_like(y[0:1])
            else:
                raise ValueError('task must be either classification or regression.')

        return {'input': x, 'decoder_input': decoder_input_data}, y
    
    if multihorizon:
        horizon = slice(0, n_horizons)

    if (task == "classification")&(alphas.size == 0):
        raise ValueError('alphas must be assigned if task is classification.')
    
    # feature_type is one of ["orderbook_features", "orderflow_features", "volume_features"]
    feature_type = model_inputs[:-1] + "_features"
    if model_inputs == "volumes_L3":
        feature_type = model_inputs[:-4] + "_features"

    # rolling window standardization
    if data_transform[:-2] == "standardize_rolling":
        TICKER = os.path.basename(files[0]).split('_')[0]
        n = int(data_transform[-1])
        aggregated_stats = pd.read_csv(os.path.join(os.path.dirname(files[0]), 'stats', TICKER + '_' + model_inputs[:-1] + '_stats.csv'), index_col=[0, 1])
        dates = aggregated_stats.index.levels[0]
        standardizations = pd.DataFrame(index = pd.MultiIndex.from_product([dates, ["mean", "std", "count"]], names=['Date', 'stat']), columns = aggregated_stats.columns)
        for i, date in enumerate(dates):
            means = aggregated_stats.xs("mean", level=1).loc[dates[i-n:i], :].copy()
            stds = aggregated_stats.xs("std", level=1).loc[dates[i-n:i], :].copy()
            counts = aggregated_stats.xs("count", level=1).loc[dates[i-n:i], :].copy()
            # total count
            count = counts.sum(axis=0)
            # aggregate mean
            mean = (means * counts).sum(axis=0) / count
            # aggregate std
            std = np.sqrt((((counts - 1)*stds**2 + counts*means**2).sum(axis=0) - count*mean**2) / (count - 1))
            standardizations.loc[(date, "mean")] = mean.values
            standardizations.loc[(date, "std")] = std.values
            standardizations.loc[(date, "count")] = count.values

    # create combined dataset
    tf_datasets = []
    for file in files:
        with np.load(file) as data:
            features = data[feature_type]
            responses = data[label][(T-1):, horizon]

        if model_inputs == "volumes":
            features = np.sum(features, axis = 2)
        mid = features.shape[1]
        if model_inputs[:7] == "volumes":
            features = features[:, (mid//2 - NF//2):(mid//2 + NF//2)]
        else:
            features = features[:, :NF]
        features = np.expand_dims(features, axis=-1)
        features = tf.convert_to_tensor(features, dtype=tf.float32)

        if task == "classification":
            if multihorizon:
                all_label = []
                for h in range(n_horizons):
                    one_label = (+1)*(responses[:, h]>=-alphas[h]) + (+1)*(responses[:, h]>alphas[h])
                    one_label = tf.keras.utils.to_categorical(one_label, 3)
                    one_label = one_label.reshape(len(one_label), 1, 3)
                    all_label.append(one_label)
                y = np.hstack(all_label)
            else:
                y = (+1)*(responses>=-alphas[horizon]) + (+1)*(responses>alphas[horizon])
                y = tf.keras.utils.to_categorical(y, 3)
        elif task == "regression":
            y = responses

        if data_transform[:11] == "standardize":
            # apply std
            date = re.search(r'\d{4}-\d{2}-\d{2}', file).group(0)
            mean = standardizations.loc[(date, "mean")].to_numpy()[:NF]
            std = standardizations.loc[(date, "std")].to_numpy()[:NF]
            features = (features - mean.reshape(-1, 1)) / std.reshape(-1, 1)         
        
        tf_datasets.append(tf.keras.preprocessing.timeseries_dataset_from_array(features, y, T, batch_size=1, sequence_stride=roll_window, shuffle=False))
    
    # tf_dataset = tf.data.Dataset.from_tensor_slices(tf_datasets).flat_map(lambda x: x)
    tf_dataset = tf_datasets[0]
    for dataset in tf_datasets[1:]:
        print(dataset)
        tf_dataset = tf_dataset.concatenate(dataset)
    
    if data_transform == "normalize":
        tf_dataset = tf_dataset.map(scale_fn)

    if multihorizon:
        tf_dataset = tf_dataset.map(add_decoder_input)

    if shuffle:
        tf_dataset = tf_dataset.shuffle(1000, reshuffle_each_iteration=False)
    else:
        tf_dataset = tf_dataset.shuffle(1, reshuffle_each_iteration=False)

    tf_dataset = tf_dataset.unbatch()
    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def CustomtfDatasetUniv(dict_of_files, NF, horizon, n_horizons, model_inputs, task, dict_of_alphas, multihorizon, data_transform, batch_size, T, roll_window, shuffle, teacher_forcing = False):
    """
    Create custom tf.dataset object to be used by model, when using multiple TICKERs with different files and alphas.
    :param dict_of_files: the files with data for each TICKER, dict of lists of strs
    :param NF: number of features, int
    :param horizon: prediction horizon, int
    :param n_horizons: number of horizons in multihorizon, int
    :param model_inputs: which input is being used "orderbooks", "orderflows", "volumes" pr "volumes_L3", str
    :param task: ML task, "regression" or "classification", str
    :param alphas: alphas for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), (tot_horizons,) array
    :param multihorizon: whether the predictions are multihorizon, if True horizon = slice(0, n_horizons), bool
    :param data_transform: transformation to apply to data, bool
                if "normalize_auto": divide by the largest value to scale between 0 & 1
                if "standardize_rolling_n": use the previous n days mean and std to standardize using data in aggregate stats
    :param T: length of lookback window for features, int
    :param batch_size: batch size for dataset, int
    :param roll_window: length of window to roll forward when extracting features/responses, int
    :param shuffle: whether to shuffle dataset, bool
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder, bool
    :return: tf_dataset: a tf.dataset object 
    """
    tf_datasets = []
    for TICKER in sorted(dict_of_files.keys()):
        files = dict_of_files[TICKER]
        alphas = dict_of_alphas[TICKER]
        tf_datasets.append(CustomtfDataset(files, 
                                           NF, 
                                           horizon, 
                                           n_horizons,
                                           model_inputs = model_inputs,
                                           task = task, 
                                           alphas = alphas, 
                                           multihorizon = multihorizon, 
                                           data_transform = data_transform,
                                           teacher_forcing = teacher_forcing, 
                                           T = T, 
                                           batch_size = batch_size, 
                                           roll_window = roll_window,
                                           shuffle = shuffle))

    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_datasets).flat_map(lambda x: x)  

    return tf_dataset