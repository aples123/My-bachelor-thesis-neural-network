import pandas as pd
import numpy as np
import json
import neural_network as n_n


def read_config(filepath):
    """
    Reads the configuration file and returns it as a JSON object.
    """
    with open(filepath) as configFile:
        config_ = json.load(configFile)
    return config_


def read_df(filename):
    """
    Reads the csv file stored in the 'resources' folder and returns the dataframe containing all columns except the first one,
    because peptide's sequence is not useful in this context.
    """
    df = pd.read_csv(f'../resources/{filename}', sep='\t')
    data = df.iloc[:, 1:]
    return data


def get_basic_data_info(data, title):
    """
    Prints the information about the dataset.
    """
    x, y = n_n.get_x_y(data)
    n_features = len(x.columns)
    print(title)
    print(f'Number of features: {n_features}')
    print(f'Number of real, biological sequences - target samples: {len(y[y == 1])}')
    print(f'Number of fake, artificial sequences - decoy samples: {len(y[y == 0])}')


def compute_qvalues(data, score_type):
    """
    Sorts the dataframe according to the given score type and calculates the False Discovery Ratio and q-values.
    Returns the dataframe with two new columns.
    """
    data_sorted = data.sort_values(score_type, ascending=False).reset_index(drop=True)
    data_sorted['FDR'] = 2 * (~(data_sorted['pos_neg'].astype(bool))).astype('uint8').cumsum(axis=0) / \
                         (data_sorted.index + 1)
    qvalues = np.zeros(len(data_sorted))
    for i in range(len(data_sorted) - 1, -1, -1):
        if i == len(data_sorted) - 1:
            qvalues[i] = data_sorted['FDR'].iloc[i]
        else:
            qvalues[i] = min(qvalues[i + 1], data_sorted['FDR'].iloc[i])
    data_sorted['qvalue'] = qvalues
    return data_sorted
