# Deep learning in order book markets: the impact of market frictions on return predictability
Repository containing the code used for the master thesis: "Deep learning in order book markets: the impact of market frictions on return predictability", by Mauricio Pereira da Silva Filho, as part of the requirements for the Msc. in Machine Learning and Data Science at Imperial College London.

The repository contains code used to:

- Prepare orderbook and message files downloaded from the LOBSTER website for further processing;
- Generate multiple types of returns;
- Train, validate and test the deep learning architectures: deepLOB, deepOF, deepVOL and DTNN;
- Analyze the results;

This repository takes large portions of the code from the papers:

- Lorenzo Lucchese, Mikko Pakkanen, and Almut Veraart. The short-term predictability
of returns in order book markets: a deep learning perspective, 2023. Git repository: [https://github.com/lorenzolucchese/deepOBs](https://github.com/lorenzolucchese/deepOBs)

- Shijie Lai, Mingxian Wang, Shengjie Zhao, and Gonzalo R. Arce. Predicting high-
frequency stock movement with differential transformer neural network. Electronics,
12(13), 2023. ISSN 2079-9292. doi: 10.3390/electronics12132943. URL [https://www.
mdpi.com/2079-9292/12/13/2943](https://mdpi.com/2079-9292/12/13/2943). Git repository: [https://github.com/j00000ker/DTNN/tree/main](https://github.com/j00000ker/DTNN/tree/main)

## Step-by-step to run

### 1. Install the required libraries

Go over the code and install all the required python libraries. One important point is: this script was run using TensorFlow version 2.5.0 and not the latest one. The script will not work with newer versions.

### 2. Download LOB data

The scripts are prepared to work with data with the format provided by [LOBSTER](https://lobsterdata.com/). The data can be downloaded through their paid subscription.

### 3. Create data root directory

The scripts expects a directory that will contain all the input and output data generated by it, say `data_root`.

The next steps follow the same numeric order as the directories in this repository.

### 4. File extraction

The script expects order book and message files to be extracted (they come zipped from LOBSTER) and put into the data root directory, in the directory `input`. The files should be divided into subdirectories for each ticker, for example `data_root/input/TICKER1`, `data_root/input/TICKER2`, etc. The notebook [pipelines/4_file_extraction/file_management.ipynb](pipelines/4_file_extraction/file_management.ipynb) helps in this step.

### 5. Data generation

This step guarantees that labelled datasets are generated using the LOB data, with multiple types of labels. This can be done by running the script [pipelines/5_data_generation/main.py](pipelines/5_data_generation/main.py). The parameters required by the script are explained in the python file. Follows an example of how to run it:

```
python3 ./pipelines/5_data_generation/main.py
--data_root data_root
--tickers MELI PANW ARM
--return_types uniform_mid_returns horizon_mid_returns pit_mid_returns 
```

### 6. Train, validate and test models

In the case of deepLOB, deepOF and deepVOL, this is done by running the script [pipelines/6_deepLOB_OF_VOL/main.py](pipelines/6_deepLOB_OF_VOL/main.py) and, in the case of DTNN, this is done by running the script [pipelines/6_DTNN/main.py](pipelines/6_DTNN/main.py). Both scripts receive the same parameters, which are detailed in the python files. Follows examples of how to run it:

```
python3 ./pipelines/6_deepLOB_OF_VOL/main.py 
--data_root data_root
--device 5 
--tickers MELI PANW ARM 
--horizons 0 8 9 
--windows 5 6 7 8 9 10 11 12 13 14
```
or

```
python3 ./pipelines/6_DTNN/main.py 
--data_root data_root
--device 5 
--tickers MELI PANW ARM 
--horizons 0 8 9 
--windows 5 6 7 8 9 10 11 12 13 14
```

The results will be generated into the directory `data_root/results`.

### 7. Analyze the results

The analysis presented in the thesis text are contained in the directory [pipelines/7_analysis](pipelines/7_analysis). The file [pipelines/7_analysis/results_consolidation.ipynb](pipelines/7_analysis/results_consolidation.ipynb) contains useful functions to compile the results generated in the previous step.

