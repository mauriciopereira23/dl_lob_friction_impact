import os
import glob
import pandas as pd
import re
from datetime import datetime
import pickle
import numpy as np
import re
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix

def multiprocess_orderbooks(TICKER, input_path, output_path, log_path, stats_path, horizons=np.array([10, 20, 30, 50, 100]), NF_volume=40, queue_depth=10, k=10, check_for_processed_data=True):
    """
    Pre-process LOBSTER data into feature-response pairs parallely. The data must be stored in the input_path
    directory as daily message book and order book files. For details of processing steps see description of process_orderbook function.
    A log file is produced tracking:
    - order book files with problems
    - message book files with problems
    - trading days with unusual open - close times
    - trading days with crossed quotes
    :param TICKER: the TICKER to be considered, str
    :param input_path: the path where the order book and message book files are stored, order book files have shape (:, 4*levels):
                       "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels", str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :param log_path: the path where processing logs are saved, str
    :param stats_path: the path where the stats for subsequent standardization are saved, str
    :param NF_volume: number of features for volume representation, only used if volume in features
               for volumes NF = 2W where W is number of ticks on each side of mid, int
               for orderbooks NF = 4*levels
               for ordeflows NF = 2*levels
    :param horizons: forecasting horizons for labels, (h,) array
    :param queue_depth: the depth beyond which to aggregate the queue for volume features, int
    :param check_for_processed_data: if True check if processed .npz files already exist in output_path and skip these dates for processing, bool 
    :return: saves the processed features in output_path, as .npz files with numpy attributes 
             "orderbook_features" (:, 4*levels) 
             "orderflow_features" (:, 2*levels) 
             "volume_features" (:, NF_volume, queue_depth)
             "mid_returns" (:, h).
    """
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))

    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]

    csv_orderbook.sort()
    csv_message.sort()

    # do not process already processed data
    if check_for_processed_data:
        npz_file_list = sorted(glob.glob(os.path.join(output_path, "*.{}".format("npz"))))
        processed_dates = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in npz_file_list]
        csv_orderbook = [file for file in csv_orderbook if re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]
        csv_message = [file for file in csv_message if re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))

    print("started multiprocessing")
    
    try:
        pool = Pool(os.cpu_count() - 2)  # leave 2 cpus free
        engine = ProcessOrderbook(TICKER, output_path, stats_path, NF_volume, queue_depth, horizons, k)
        logs = pool.map(engine, csv_orderbook)
    except Exception as e:
        print(f"Caught an exception: {e}")
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    print("finished multiprocessing")

    # with open(log_path + "/processing_logs.txt", "w") as f:
    #     for log in logs:
    #         f.write(log + "\n")

    print("please check processing logs.")


class ProcessOrderbook(object):
    """
    Multiprocessing engine for pre-processing LOBSTER data into feature-response npz files.
    """
    def __init__(self, TICKER, output_path, stats_path, NF_volume, queue_depth, horizons, k):
        self.TICKER = TICKER
        self.output_path = output_path
        self.stats_path = stats_path
        self.NF_volume = NF_volume
        self.queue_depth = queue_depth
        self.horizons = horizons
        self.k = k

    def __call__(self, orderbook_name):
        output = process_orderbook(orderbook_name, self.TICKER, self.output_path, self.stats_path, self.NF_volume, self.queue_depth, self.horizons, self.k)
        return output


def process_orderbook(orderbook_name, TICKER, output_path, stats_path, NF_volume, queue_depth, horizons, k):
    """
    Function carrying out processing of single order book orderbook_name. Features are not normalized.
    The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 10 minutes of market activity (inside usual opening times) are dropped.
    - the smoothed returns at the requested horizons (in order book changes) are returned
      if smoothing = "horizon": l = (m+ - m)/m, where m+ denotes the mean of the next h mid-prices, m(.) is current mid price.
      if smoothing = "uniform": l = (m+ - m)/m, where m+ denotes the mean of the k+1 mid-prices centered at m(. + h), m(.) is current mid price.
    :param orderbook_name: order book to process, of shape (T, 4*levels):
                          "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels",
                          str
    :param TICKER: the TICKER to be considered, str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :param stats_path: the path where the stats for subsequent standardization are saved, str
    :param NF_volume: number of features for volume representation, only used if volume in features
               for volumes NF = 2W where W is number of ticks on each side of mid, int
               for orderbooks NF = 4*levels
               for ordeflows NF = 2*levels
    :param queue_depth: the depth beyond which to aggregate the queue for volume features, int
    :param horizons: forecasting horizons for labels, (h,) array
    :return: saves the processed features in output_path, as .npz files with numpy attributes 
             "orderbook_features" (:, 4*levels) 
             "orderflow_features" (:, 2*levels) 
             "volume_features" (:, NF_volume, queue_depth)
             "responses" (:, h).
    """
    print(orderbook_name)
    log = ''

    ### load LOBSTER orderbook and message files
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    levels = int(df_orderbook.shape[1] / 4)
    feature_names_raw = ["ASKp", "ASKs", "BIDp", "BIDs"]
    orderbook_feature_names = []
    for i in range(1, levels + 1):
        for j in range(4):
            orderbook_feature_names += [feature_names_raw[j] + str(i)]
    df_orderbook.columns = orderbook_feature_names

    # compute the mid prices
    df_orderbook.insert(0, "mid price", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2)

    # get date
    match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
    date = datetime.strptime(match, "%Y-%m-%d")

    # read times from message file. keep a record of problematic files
    message_name = orderbook_name.replace("orderbook", "message")
    try:
        df_message = pd.read_csv(message_name, usecols=[0, 1, 2, 3, 4, 5], header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read messagebook.'

    # check the two df have the same length
    assert (len(df_message) == len(df_orderbook))

    # add column names to message book
    df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

    ### process potential data anomalies

    # 1. remove trading halts
    trading_halts_start = df_message[(df_message["event type"] == 7)&(df_message["price"] == -1)].index
    trading_halts_end = df_message[(df_message["event type"] == 7)&(df_message["price"] == 1)].index
    trading_halts_index = np.array([])
    for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
        trading_halts_index = np.append(trading_halts_index, df_message.index[(df_message.index >= halt_start)&(df_message.index < halt_end)])
    if len(trading_halts_index) > 0:
        for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
            log = log + 'Warning: trading halt between ' + str(df_message.loc[halt_start, "seconds"]) + ' and ' + str(df_message.loc[halt_end, "seconds"]) + ' in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(trading_halts_index)
    df_message = df_message.drop(trading_halts_index)

    # 2. remove crossed quotes
    crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
    if len(crossed_quotes_index) > 0:
        log = log + 'Warning: ' + str(len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(crossed_quotes_index)
    df_message = df_message.drop(crossed_quotes_index)

    # add the seconds since midnight column to the order book from the message book
    df_orderbook.insert(0, "seconds", df_message["seconds"])

    df_orderbook_full = df_orderbook
    df_message_full = df_message

    # 3. one conceptual event (e.g. limit order modification which is implemented as a cancellation followed
    # by an immediate new arrival, single market order executing against multiple resting limit orders) may
    # appear as multiple rows in the message file, all with the same timestamp.
    # We hence group the order book data by unique timestamps and take the last entry.
    df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)
    df_message = df_message.groupby(["seconds"]).tail(1)

    # 4. check market opening times for strange values
    market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60  # open at minute before first transaction
    market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # close at minute after last transaction

    if not (market_open == 9.5 and market_close == 16):
        log = log + 'Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(market_close) + '.\n'
    
    # drop values outside of market hours
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                    (df_orderbook["seconds"] <= 57600)]
    df_message = df_message.loc[(df_message["seconds"] >= 34200) &
                                (df_message["seconds"] <= 57600)]
    
    ### compute orderflow features
    ASK_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains('ASKp')]
    BID_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains('BIDp')]
    ASK_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains('ASKs')]
    BID_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains('BIDs')]

    ASK_price_changes = ASK_prices.diff().dropna().to_numpy()
    BID_price_changes = BID_prices.diff().dropna().to_numpy()
    ASK_size_changes = ASK_sizes.diff().dropna().to_numpy()
    BID_size_changes = BID_sizes.diff().dropna().to_numpy()

    ASK_sizes = ASK_sizes.to_numpy()
    BID_sizes = BID_sizes.to_numpy()

    ASK_OF = (ASK_price_changes > 0.0) * (-ASK_sizes[:-1, :]) + (ASK_price_changes == 0.0) * ASK_size_changes + (ASK_price_changes < 0) * ASK_sizes[1:, :]
    BID_OF = (BID_price_changes < 0.0) * (-BID_sizes[:-1, :]) + (BID_price_changes == 0.0) * BID_size_changes + (BID_price_changes > 0) * BID_sizes[1:, :]

    # remove all price-volume features and add in orderflow
    df_orderflow = pd.DataFrame([], index = df_orderbook.index[1:])
    feature_names_raw = ["ASK_OF", "BID_OF"]
    orderflow_feature_names = []
    for feature_name in feature_names_raw:
        for i in range(1, levels + 1):
            orderflow_feature_names += [feature_name + str(i)]
    df_orderflow[orderflow_feature_names] = np.concatenate([ASK_OF, BID_OF], axis=1)
    
    # re-order columns
    feature_names_reordered = [[]]*len(orderflow_feature_names)
    feature_names_reordered[::2] = orderflow_feature_names[:levels]
    feature_names_reordered[1::2] = orderflow_feature_names[levels:]
    orderflow_feature_names = feature_names_reordered

    df_orderflow = df_orderflow[orderflow_feature_names]

    # 6. drop first and last 10 minutes of trading using seconds (do this after orderflow computation)
    market_open_seconds = market_open * 60 * 60 + 10 * 60
    market_close_seconds = market_close * 60 * 60 - 10 * 60
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                    (df_orderbook["seconds"] <= market_close_seconds)]

    df_message = df_message.loc[(df_message["seconds"] >= market_open_seconds) &
                                (df_message["seconds"] <= market_close_seconds)]

    df_orderflow = df_orderflow.loc[df_orderbook.index]
    
    ### compute volume features
    volumes = np.zeros((len(df_orderbook_full), NF_volume, queue_depth))
    
    # Assumes tick_size = 0.01 $, as per LOBSTER data
    ticks = np.hstack((np.outer(np.round((df_orderbook_full["mid price"] - 25) / 100) * 100, np.ones(NF_volume//2)) + 100 * np.outer(np.ones(len(df_orderbook_full)), np.arange(-NF_volume//2+1, 1)),
                        np.outer(np.round((df_orderbook_full["mid price"] + 25) / 100) * 100, np.ones(NF_volume//2)) + 100 * np.outer(np.ones(len(df_orderbook_full)), np.arange(NF_volume//2))))
    ticks = ticks.astype(int)

    orderbook_states = df_orderbook_full[orderbook_feature_names]
    orderbook_states_prices = orderbook_states.values[:, ::2]
    orderbook_states_volumes = orderbook_states.values[:, 1::2]

    # volumes = np.zeros((len(df_orderbook_full), NF_volume))

    # for i in range(NF_volume):
    #     # match tick prices with prices in levels of orderbook
    #     flags = (orderbook_states_prices == np.repeat(ticks[:, i].reshape((len(orderbook_states_prices), 1)), orderbook_states_prices.shape[1], axis=1))
    #     volumes[flags.sum(axis=1) > 0, i] = orderbook_states_volumes[flags]

    prices_dict = {}
    
    # skip first message_df row
    skip = True
    
    for i, index in enumerate(df_message_full.index):
        seconds = df_message_full.loc[index, "seconds"]
        event_type = df_message_full.loc[index, "event type"]
        order_id = df_message_full.loc[index, "order ID"]
        price = df_message_full.loc[index, "price"]
        volume = df_message_full.loc[index, "volume"]

        # if new price is entering range (re-)initialize dict
        for price_ in orderbook_states_prices[i, :]:
            if (price_ in orderbook_states_prices[i - 1, :]) and (price_ in prices_dict.keys()):
                pass
            else:
                j = np.where(orderbook_states_prices[i, :]==price_)[0][0]
                volume_ = orderbook_states_volumes[i, j]
                prices_dict[price_] = pd.DataFrame(np.array([[volume_, 0]]), index = ['id'], columns = ["volume", "seconds"])
                if price_ == price:
                    skip = True
        
        price_df = prices_dict.get(price, pd.DataFrame([], columns = ["volume", "seconds"], dtype=float))
        
        # if new price also corresponds to message_df price skip to avoid double counting
        if skip:
            skip = False
            pass
        
        # if the price from message_df is not in df_orderbook prices, i.e. a failure in LOBSTER reconstruction system, treat as hidden market order
        elif (not price in orderbook_states_prices[(i-1):(i+1), :])&(event_type in [1, 2, 3, 4, 5]):
            event_type = 5
            pass

        # new limit order
        elif event_type == 1:
            price_df.loc[order_id] = [volume, seconds]
        
        # cancellation (partial deletion)
        elif event_type == 2:
            # if order_id is not in price dataframe then this is part of initial order
            if order_id in price_df.index:
                price_df.loc[order_id, "volume"] -= volume
            else:
                price_df.loc["id", "volume"] -= volume
        
        # deletion
        elif event_type == 3:
            # if id is not present (i.e. it is at the start of order book), delete from "id" and check if "id" is empty
            if order_id in price_df.index:
                price_df = price_df.drop(order_id)
            else:
                price_df.loc["id", "volume"] -= volume
                if price_df.loc["id", "volume"] == 0:
                    price_df = price_df.drop("id")
        
        # execution of visible limit order
        elif event_type == 4:
            if order_id in price_df.index:
                price_df.loc[order_id, "volume"] -= volume
                if price_df.loc[order_id, "volume"] == 0:
                    price_df = price_df.drop(order_id)
            else:
                price_df.loc["id", "volume"] -= volume
                if price_df.loc["id", "volume"] == 0:
                    price_df = price_df.drop("id")
        
        # execution of hidden limit order
        elif event_type == 5:
            pass
        
        # auction trade
        elif event_type == 6:
            pass
        
        # trading halt
        elif event_type == 7:
            # re-initialize prices_dict
            prices_dict = {}
            for price_ in orderbook_states_prices[i, :]:
                j = np.where(orderbook_states_prices[i, :]==price_)[0][0]
                volume_ = orderbook_states_volumes[i, j]
                prices_dict[price_] = pd.DataFrame(np.array([[volume_, 0]]), index = ['id'], columns = ["volume", "seconds"])
            pass
        
        else:
            raise ValueError("LOBSTER event type must be 1, 2, 3, 4, 5, 6 or 7")
        
        price_df = price_df.sort_values(by="seconds")
        prices_dict[price] = price_df

        # update orderbooks_L3
        if event_type in [5, 6]:
            volumes[i, :, :] = volumes[i - 1, :, :] 
        elif (ticks[i, :] == ticks[i - 1, :]).all() & (not event_type == 7):
            if price in ticks[i, :]:
                j = np.where(ticks[i, :] == price)[0][0]
                volumes[i, :, :] = volumes[i - 1, :, :] 
                if len(price_df) == 0:
                    volumes[i, j, :] = np.zeros(queue_depth)
                elif len(price_df) < queue_depth:
                    volumes[i, j, :len(price_df)] = price_df["volume"].values
                    volumes[i, j, len(price_df):] = 0
                else:
                    volumes[i, j, :(queue_depth-1)] = price_df["volume"].values[:(queue_depth-1)]
                    volumes[i, j, queue_depth-1] = np.sum(price_df["volume"].values[(queue_depth-1):])
            else:
                volumes[i, :, :] = volumes[i - 1, :, :]
        else:
            for j, price_ in enumerate(ticks[i, :]):
                price_df_ = prices_dict.get(price_, [])
                if len(price_df_) == 0:
                    volumes[i, j, :] = np.zeros(queue_depth)
                elif len(price_df_) < queue_depth:
                    volumes[i, j, :len(price_df_)] = price_df_["volume"].values
                else:
                    volumes[i, j, :(queue_depth-1)] = price_df_["volume"].values[:(queue_depth-1)]
                    volumes[i, j, queue_depth-1] = np.sum(price_df_["volume"].values[(queue_depth-1):])
        
    # need then to remove all same timestamps (collapse to last) and first/last 10 minutes
    volumes = volumes[df_orderbook_full.index.isin(df_orderbook.index), :, :]

    ### compute returns
    # create labels for returns with smoothing labelling method
    uniform_rolling_mid = df_orderbook["mid price"].rolling(k+1, center=True).mean()
    uniform_rolling_mid = uniform_rolling_mid.to_numpy().flatten()
    for h in horizons:
        # uniform smoothing.
        uniform_smooth_pct_change = uniform_rolling_mid[h:] / df_orderbook["mid price"][:-h] - 1
        df_orderbook[f"uniform_mid_{h}"] = uniform_smooth_pct_change

        # Horizon smoothing.
        horizon_rolling_mid = df_orderbook["mid price"].rolling(h).mean().dropna()[1:]
        horizon_rolling_mid = horizon_rolling_mid.to_numpy().flatten()
        horizon_smooth_pct_change = horizon_rolling_mid / df_orderbook["mid price"][:-h] - 1
        df_orderbook[f"horizon_mid_{h}"] = np.concatenate((horizon_smooth_pct_change, np.repeat(np.NaN, int(h))))

        # Point-in-time return.
        shifted_mid = df_orderbook["mid price"].shift(h)
        pit_pct_change = shifted_mid[h:] / df_orderbook["mid price"][:-h] - 1
        df_orderbook[f"pit_mid_{h}"] = pit_pct_change

        # Bid-ask return
        df_orderbook[f"ask_bid_{h}"] = df_orderbook["BIDp1"].shift(h) / df_orderbook["ASKp1"] - 1

        # Ask-bid return
        df_orderbook[f"bid_ask_{h}"] = -df_orderbook["ASKp1"].shift(h) / df_orderbook["BIDp1"] - 1

        # Latency return
        latencies_s = np.array([30, 300, 3000])
        latencies_mus = latencies_s / 1e6

        for i, latency in enumerate(latencies_mus):
            seconds = df_orderbook["seconds"]
            seconds_delayed = seconds + latency
            indices = seconds.searchsorted(seconds_delayed, side='right') - 1

            mid_prices = df_orderbook["mid price"]
            ask_prices = df_orderbook["ASKp1"]
            bid_prices = df_orderbook["BIDp1"]


            df_orderbook[f"delayed_mid_price_{latencies_s[i]}"] = mid_prices.iloc[indices].values
            df_orderbook[f"delayed_ask_price_{latencies_s[i]}"] = ask_prices.iloc[indices].values
            df_orderbook[f"delayed_bid_price_{latencies_s[i]}"] = bid_prices.iloc[indices].values

            shifted_delayed_mid = df_orderbook[f"delayed_mid_price_{latencies_s[i]}"].shift(h)
            delayed_mid_change = shifted_delayed_mid[h:] / df_orderbook[f"delayed_mid_price_{latencies_s[i]}"][:-h] - 1
            df_orderbook[f"delayed_mid_{latencies_s[i]}_{h}"] = delayed_mid_change

            shifted_delayed_bid = df_orderbook[f"delayed_bid_price_{latencies_s[i]}"].shift(h)
            delayed_bid_ask_change = shifted_delayed_bid[h:] / df_orderbook[f"delayed_ask_price_{latencies_s[i]}"][:-h] - 1
            df_orderbook[f"delayed_ask_bid_{latencies_s[i]}_{h}"] = delayed_bid_ask_change

            shifted_delayed_ask = df_orderbook[f"delayed_ask_price_{latencies_s[i]}"].shift(h)
            delayed_ask_bid_change = -shifted_delayed_ask[h:] / df_orderbook[f"delayed_bid_price_{latencies_s[i]}"][:-h] - 1
            df_orderbook[f"delayed_bid_ask_{latencies_s[i]}_{h}"] = delayed_ask_bid_change

    ### drop seconds and mid price columns
    timestamps = df_orderbook["seconds"]
    df_orderbook = df_orderbook.drop(["seconds", "mid price"], axis=1)

    uniform_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"uniform_mid_{h}" for h in horizons]].values
    df_orderbook.drop([f"uniform_mid_{h}" for h in horizons], axis=1, inplace=True)

    horizon_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"horizon_mid_{h}" for h in horizons]].values
    df_orderbook.drop([f"horizon_mid_{h}" for h in horizons], axis=1, inplace=True)

    pit_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"pit_mid_{h}" for h in horizons]].values
    df_orderbook.drop([f"pit_mid_{h}" for h in horizons], axis=1, inplace=True)

    bid_ask_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"bid_ask_{h}" for h in horizons]].values
    df_orderbook.drop([f"bid_ask_{h}" for h in horizons], axis=1, inplace=True)

    ask_bid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"ask_bid_{h}" for h in horizons]].values
    df_orderbook.drop([f"ask_bid_{h}" for h in horizons], axis=1, inplace=True)

    latency_30_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_mid_30_{h}"for h in horizons]].values
    latency_300_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_mid_300_{h}"for h in horizons]].values
    latency_3000_mid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_mid_3000_{h}"for h in horizons]].values

    latency_30_bid_ask_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_bid_ask_30_{h}"for h in horizons]].values
    latency_300_bid_ask_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_bid_ask_300_{h}"for h in horizons]].values
    latency_3000_bid_ask_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_bid_ask_3000_{h}"for h in horizons]].values

    latency_30_ask_bid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_ask_bid_30_{h}"for h in horizons]].values
    latency_300_ask_bid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_ask_bid_300_{h}"for h in horizons]].values
    latency_3000_ask_bid_returns = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2),:].loc[:,[f"delayed_ask_bid_3000_{h}"for h in horizons]].values


    orderbook_features = df_orderbook.iloc[max(horizons):-(max(horizons)+k//2), :]
    orderflow_features = df_orderflow.iloc[max(horizons):-(max(horizons)+k//2), :]
    volume_features = volumes[max(horizons):-(max(horizons)+k//2), :, :]
    
    # save
    output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + str(date.date()))
    np.savez(output_name + ".npz",
             timestamps=timestamps,
             orderbook_features=orderbook_features.values,
             orderflow_features=orderflow_features.values,
             volume_features=volume_features,
             uniform_mid_returns=uniform_mid_returns,
             horizon_mid_returns=horizon_mid_returns,
             pit_mid_returns=pit_mid_returns,
             bid_ask_returns=bid_ask_returns,
             ask_bid_returns=ask_bid_returns,
             latency_30_mid_returns=latency_30_mid_returns,
             latency_300_mid_returns=latency_300_mid_returns,
             latency_3000_mid_returns=latency_3000_mid_returns,
             latency_30_bid_ask_returns=latency_30_bid_ask_returns,
             latency_300_bid_ask_returns=latency_300_bid_ask_returns,
             latency_3000_bid_ask_returns=latency_3000_bid_ask_returns,
             latency_30_ask_bid_returns=latency_30_ask_bid_returns,
             latency_300_ask_bid_returns=latency_300_ask_bid_returns,
             latency_3000_ask_bid_returns=latency_3000_ask_bid_returns
            )

    ### save mean, standard deviation and count for rolling window normalisation
    orderbook_stats = pd.DataFrame([orderbook_features.mean(axis=0), orderbook_features.std(axis=0), orderbook_features.count(axis=0)], index = ['mean', 'std', 'count'])
    orderflow_stats = pd.DataFrame([orderflow_features.mean(axis=0), orderflow_features.std(axis=0), orderflow_features.count(axis=0)], index = ['mean', 'std', 'count'])

    orderbook_stats.to_csv(os.path.join(stats_path, TICKER + '_orderbook_stats_' + str(date.date()) + '.csv'))
    orderflow_stats.to_csv(os.path.join(stats_path, TICKER + '_orderflow_stats_' + str(date.date()) + '.csv'))

    return log + orderbook_name + ' completed.'

def aggregate_stats(TICKER, stats_path, features=["orderbook", "orderflow"]):
    """
    Function for aggregating processed features (e.g. orderbook and orderflow)
    :param TICKER: the TICKER to be considered, str
    :param stats_path: the path where daily stats are saved, str
    :param features: features for which to aggregate stats, list of str
    """
    csv_file_list = glob.glob(os.path.join(stats_path, "*.{}".format("csv")))

    for feature in features:
        feature_stats = {datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', name).group(), '%Y-%m-%d').date(): pd.read_csv(name, index_col=0) for name in csv_file_list if feature in name and re.search(r'\d{4}-\d{2}-\d{2}', name) is not None}
    
        feature_stats = dict(sorted(feature_stats.items()))
    
        aggregated_feature_stats = pd.concat(feature_stats, names=['Date'])

        aggregated_feature_stats.index = aggregated_feature_stats.index.rename('stat', level=1)

        aggregated_feature_stats.to_csv(os.path.join(stats_path, TICKER + '_' + feature + '_stats.csv'))
        
        
def percentiles_features(TICKER, processed_data_path, stats_path, percentiles, features=["orderbook", "orderflow", "volume"], levels = 10, NF_volume = 40):
    """
    Function for summarizing percentiles of features once data has been processed.
    :param TICKER: the TICKER to be considered, str
    :param processed_data_path: the path where the processed data is stored, str
    :param stats_path: the path where stats are to be saved, str
    :param percentiles: the percentiles to be computed, list or np.array
    :param features: features for which to compute daily stats, list of str
    :param levels: number of levels which are stored in the npz files, int
    :param NF_volume: number of features for volume representation, only used if volume in features
    """
    npz_file_list = sorted(glob.glob(os.path.join(processed_data_path, "*.{}".format("npz"))))
    
    for feature in features:
        # add in feature names
        feature_names = []
        if feature == "orderbook":
            feature_names_raw = ["ASKp", "ASKs", "BIDp", "BIDs"]
            for i in range(1, levels + 1):
                for j in range(4):
                    feature_names += [feature_names_raw[j] + str(i)]
        elif feature == "orderflow":
            feature_names_raw = ["ASK_OF", "BID_OF"]
            for i in range(1, levels + 1):
                for feature_name in feature_names_raw:
                    feature_names += [feature_name + str(i)]
        elif feature == "volume":
            queue_depths_names = []
            for i in range(NF_volume//2, 0, -1):
                feature_names += ["BIDv" + str(i)]
                queue_depths_names += ["BIDq" + str(i)]
            for i in range(1, NF_volume//2 + 1):
                feature_names += ["ASKv" + str(i)]
                queue_depths_names += ["ASKq" + str(i)]
        
        daily_stats_dfs = {}
        feature_matrix_all = np.array([]).reshape(0, len(feature_names))
        if feature == "volume":
            queue_depths_all = np.array([]).reshape(0, len(queue_depths_names))
            daily_queue_depths_stats_dfs = {}

        for file in npz_file_list:
            date = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', file).group(), '%Y-%m-%d').date()
            print(date)
            with np.load(file) as data:
                feature_matrix = data[feature + "_features"]
            try:
                if feature == "volume":
                    # first compute stats related to queue depth
                    queue_depths = (feature_matrix > 0).sum(axis=-1)
                    percentiles_queue_depths = np.percentile(queue_depths, percentiles, axis=0)
                    daily_queue_depths_stats_dfs[date]= pd.DataFrame(percentiles_queue_depths, index = percentiles, columns = queue_depths_names)
                    queue_depths_all = np.concatenate([queue_depths_all, queue_depths], axis=0)
                    # then aggregate volumes to apply quartile stats as for orderbook and orderflow
                    feature_matrix = feature_matrix.sum(axis=-1)
                percentiles_features = np.percentile(feature_matrix, percentiles, axis=0)
                daily_stats_dfs[date]= pd.DataFrame(percentiles_features, index = percentiles, columns = feature_names)
                feature_matrix_all = np.concatenate([feature_matrix_all, feature_matrix], axis=0)
            except:
                print('The following date was skipped: ' + date.strftime("%d-%m-%Y"))
                continue
        percentiles_features_all = np.percentile(feature_matrix_all, percentiles, axis=0)
        daily_stats_dfs["all"] = pd.DataFrame(percentiles_features_all, index = percentiles, columns = feature_names)
        stats_df = pd.concat(daily_stats_dfs, names = ['Date'])
        stats_df.to_csv(os.path.join(stats_path, TICKER + '_' + feature + '_percentiles.csv'))
        if feature == "volume":
            percentiles_queue_depths_all = np.percentile(queue_depths_all, percentiles, axis=0)
            daily_queue_depths_stats_dfs["all"] = pd.DataFrame(percentiles_queue_depths_all, index = percentiles, columns = queue_depths_names)
            queue_depths_stats_df = pd.concat(daily_queue_depths_stats_dfs, names = ['Date'])
            queue_depths_stats_df.to_csv(os.path.join(stats_path, TICKER + '_queue_depth_percentiles.csv'))

def dependence_responses(TICKER, processed_data_path, results_path, stats_path, horizons=[10, 20, 30, 50, 100, 200, 300, 500, 1000], k=10):
    """
    Function for summarizing dependence of responses.
    :param TICKER: the TICKER to be considered, str
    :param processed_data_path: the path where the processed data is stored, str
    :param results_path: the path where the results are stored, str
    :param stats_path: the path where stats are to be saved, str
    :param horizons: the horizons at which the responses are defined, list/np.array
    :param k: smoothing window for averaging prices in return definition, int
    """
    npz_file_list = sorted(glob.glob(os.path.join(processed_data_path, "*.{}".format("npz"))))

    daily_dfs = {}
    for w in range(11):
        with open(os.path.join(results_path, TICKER, "W" + str(w), "alphas.pkl"), 'rb') as f:
            alphas = pickle.load(f)
        with open(os.path.join(results_path, TICKER, "W" + str(w), "val_train_test_dates.pkl"), 'rb') as f:
            dates = pickle.load(f)
        npz_file_list_window = [file for file in npz_file_list if re.search(r'\d{4}-\d{2}-\d{2}', file).group() in dates[0] + dates[1] + dates[2]]
        for file in npz_file_list_window:
            date = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', file).group(), '%Y-%m-%d').date()
            print(date)
            with np.load(file) as data:
                responses = data['mid_returns']
            horizon_dfs = {}
            for h, horizon in enumerate(horizons):
                labels = (+1)*(responses[:, h]>=-alphas[h]) + (+1)*(responses[:, h]>alphas[h]) - 1
                horizon_dfs[horizon] = pd.DataFrame(confusion_matrix(labels[:-horizon-k//2], labels[horizon+k//2:]), index = ["down", "stationary", "up"], columns = ["down", "stationary", "up"])
            daily_dfs[date] = pd.concat(horizon_dfs, names = ['horizon'])
    full_df = pd.concat(daily_dfs, names = ['Date'])
    full_df.to_csv(os.path.join(stats_path, TICKER + '_dependence_responses.csv'))
    

if __name__ == "__main__":
    dependence_responses("LILAK", 
                         "data/LILAK", 
                         "results",
                         "data/stats")