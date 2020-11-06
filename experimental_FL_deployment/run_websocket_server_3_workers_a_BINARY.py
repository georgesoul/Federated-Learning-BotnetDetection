from multiprocessing import Process
import argparse
import os
import logging

import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
from syft.frameworks.torch.fl import utils

import torch
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

KEEP_LABELS_DICT = {
    "alice": ['ramnit', 'kraken', 'simda'],
    "bob": ['banjori', 'pykspa', 'ramdo', 'qakbot', 'cryptolocker'],
    "charlie": ['locky', 'corebot', 'dircrypt'],
    "testing": ['pykspa', 'simda', 'kraken', 'qakbot', 'cryptolocker', 'locky', 'ramdo', 'ramnit', 'banjori', 'corebot', 'dircrypt']
}

original = ['benign']
dga = ['pykspa', 'simda', 'kraken', 'qakbot', 'cryptolocker', 'locky', 'ramdo', 'ramnit', 'banjori', 'corebot', 'dircrypt']

DATA_FILE = './data/traindata.pkl'

def init_data(id, keep_labels):
    data = pickle.load(open(DATA_FILE, 'rb'))
    test_dataset=data[:21000]
    test_dataset_dga = test_dataset.loc[test_dataset['label'].isin(keep_labels)]
    test_dataset_benign = test_dataset.loc[test_dataset['label'].isin(original)]
    test_dataset = pd.concat([test_dataset_dga, test_dataset_benign], ignore_index=True, sort=False)
    data = data[21000:]
    X = data["text"]
    labels = data["label"]
    data_client = data.loc[data['label'].isin(keep_labels)]
    data_benign = data.loc[data['label'].isin(original)]

    if id=='alice':
        data_client = pd.concat([data_client, data_benign[:30000]], ignore_index=True, sort=False)
    elif id=='bob':
        data_client = pd.concat([data_client, data_benign[30000:65000]], ignore_index=True, sort=False)
    elif id=='charlie':
        data_client = pd.concat([data_client, data_benign[65000:]], ignore_index=True, sort=False)
    else:
        data_client=test_dataset
    y = [0 if x == 'benign' else 1 for x in data_client['label']]
    #valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    valid_chars = {'7': 1, 'i': 2, 'x': 3, '5': 4, 'w': 5, 't': 6, 'v': 7, 'g': 8, 'k': 9, 'd': 10, 'z': 11, '6': 12, '-': 13, '_': 14, 'a': 15,
    'p': 16, 'e': 17, '9': 18, 'b': 19, 'f': 20, 'y': 21, '2': 22, 'c': 23, 'l': 24, 's': 25, 'n': 26, 'h': 27, '3': 28, 'u': 29, 'm': 30, '0': 31,
    'r': 32, 'j': 33, '8': 34, 'o': 35, '4': 36, '1': 37, 'q': 38}
    max_features = len(valid_chars) + 1
    max_len = np.max([len(x) for x in X])
    X = data_client["text"]
    X = [[valid_chars[y] for y in x] for x in X]
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='pre')
    return np.array(X), np.array(y).reshape(len(y),1), max_features, max_len



def start_websocket_server_worker(
    id, host, port, hook, verbose, keep_labels=None, training=True, pytest_testing=False
):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = WebsocketServerWorker(id=id, host=host, port=port, hook=hook, verbose=verbose)

    X, Y, max_features, max_len = init_data(id, keep_labels)
    X, x_test, Y, y_test = train_test_split(X, Y, test_size=0.0001, shuffle=True)

    if not training:
        selected_data = torch.LongTensor(X)
        selected_targets = torch.LongTensor(Y).squeeze(1)
    else:
        if id=='alice':
            selected_data = torch.LongTensor(X)
            selected_targets = torch.LongTensor(Y).squeeze(1)
        elif id=='bob':
            selected_data = torch.LongTensor(X)
            selected_targets = torch.LongTensor(Y).squeeze(1)
        elif id=='charlie':
            selected_data = torch.LongTensor(X)
            selected_targets = torch.LongTensor(Y).squeeze(1)

    if training:


        dataset = sy.BaseDataset(
            data=selected_data, targets=selected_targets
        )
        key = "dga"
    else:
        dataset = sy.BaseDataset(
            data=selected_data,
            targets=selected_targets,
        )
        key = "dga_testing"

    # Adding Dataset
    server.add_dataset(dataset, key=key)

    count = [0] * 2

    for i in range(2):
        count[i] = (dataset.targets == i).sum().item()
        logger.info("      %s: %s", i, count[i])

    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("Examples in local dataset: %s", len(server.datasets[key]))

    server.start()
    return server


if __name__ == "__main__":

    # Logging setup
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""if set, websocket server worker will be started in verbose mode""",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="normal",
    )

    parser.add_argument("--pytest_testing", action="store_true", help="""Used for pytest testing""")
    args = parser.parse_args()

    hook = sy.TorchHook(torch)

    if args.notebook == "cnn-parallel" or args.pytest_testing == True:
        server = start_websocket_server_worker(
            id=args.id,
            host=args.host,
            port=args.port,
            hook=hook,
            verbose=args.verbose,
            keep_labels=KEEP_LABELS_DICT[args.id]
            if args.id in KEEP_LABELS_DICT
            else list(range(2)),
            training=not args.testing,
        )
