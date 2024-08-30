import argparse
from data_process import multiprocess_orderbooks, aggregate_stats
import numpy as np
import pandas as pd
import time
import os

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
        "--tickers",
        type=str,
        nargs='+',
        required = True,
        help = "(list[str]) List of tickers.",
    )
    parser.add_argument(
        "--return_types",
        type=str,
        nargs='+',
        required = True,
        help = """(list[str]) List of the types of returns to calculate. Can be 'uniform_mid_returns', 
        'horizon_mid_returns', 'pit_mid_returns', 'bid_ask_returns', 'ask_bid_returns', 'latency_30_mid_returns',
        'latency_300_mid_returns', 'latency_3000_mid_returns', 'latency_30_bid_ask_returns',
        'latency_300_bid_ask_returns', 'latency_3000_bid_ask_returns', 'latency_30_ask_bid_returns',
        'latency_300_ask_bid_returns' and 'latency_3000_ask_bid_returns'""",
    )
    args = vars(parser.parse_args())
    return args

def main():
    params = read_args()
    print("------------------------- Creating pre-processed data -------------------------")
    ROOT_DIR = params["data_root"]
    for TICKER in params["tickers"]:
        input_path = os.path.join(ROOT_DIR, "input", TICKER)
        log_path = os.path.join(ROOT_DIR, "data", "logs", TICKER + "_processing_logs")
        horizons = np.array([10, 20, 30, 50, 100, 200, 300, 500, 1000, 10000])
        
        os.makedirs(log_path, exist_ok=True)
        
        # ============================================================================
        # LOBSTER DATA (multiprocess)
        
        output_path = os.path.join(ROOT_DIR, "data", TICKER)
        os.makedirs(output_path, exist_ok=True)
        stats_path = os.path.join(output_path, "stats")
        os.makedirs(stats_path, exist_ok=True)
        
        startTime = time.time()
        multiprocess_orderbooks(TICKER=TICKER,
                                input_path=input_path, 
                                output_path=output_path,
                                log_path=log_path, 
                                stats_path=stats_path,
                                horizons=horizons, 
                                NF_volume=40, 
                                queue_depth=10, 
                                k=10,
                                check_for_processed_data=True)
        executionTime = (time.time() - startTime)
        
        print("Execution time in minutes: " + str(executionTime/60))
        
        aggregate_stats(TICKER, stats_path)

    print("------------------------- Creating returns files -------------------------")
    data_dir = os.path.join(ROOT_DIR,"data")
    for TICKER in params["tickers"]:
        ticker_dir = os.path.join(data_dir, TICKER)
        all_files = os.listdir(ticker_dir)
        npz_files = [os.path.join(ticker_dir, file) for file in all_files if file.endswith(".npz")]
        for return_type in params["return_types"]:
            print(return_type)
            output_dir = os.path.join(ticker_dir, return_type)
            os.makedirs(output_dir, exist_ok=True)
            for i, npz_file in enumerate(npz_files):
                data = np.load(npz_file)
                features_returns_df = pd.DataFrame(data[return_type])
                date = npz_files[i].split("_")[-1].split(".")[0]
                output_path = os.path.join(output_dir, date + ".csv")
                print(output_path)
                features_returns_df.to_csv(output_path, header=True, index=False)

if __name__ == "__main__":
    main()