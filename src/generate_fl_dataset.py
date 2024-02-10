

import os
import errno
import argparse
import sys
import pickle
import json

import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# import tensorflow as tf

# from data_utils import generate_alignment_data, load_CIFAR10_data
# from data_utils import load_CIFAR100_data, generate_partial_data, generate_bal_private_data
# from data_utils import load_FEMNIST_data, load_MNIST_data, load_ready_data
from utility import read_config 

private_dataset_name = 'CIFAR10' # 'CIFAR10', 'CIFAR100', 'FEMNIST', 'MNIST
if private_dataset_name in ["CIFAR10", "CIFAR100"]:
    public_dataset_name = 'CIFAR10' if private_dataset_name == 'CIFAR100' else 'CIFAR100'
    conf_file = os.path.abspath("../conf/CIFAR_balance_conf.json")
else :
    public_dataset_name = 'MNIST' if private_dataset_name == 'FEMNIST' else 'FEMNIST'
    conf_file = os.path.abspath("../conf/EMNIST_balance_conf.json")

# configs = read_config(conf_file)
# N_parties = 20 
# N_samples_per_class = 100


# data_load_functions = {
#     "MNIST": load_MNIST_data,
#     "FEMNIST": load_FEMNIST_data,
#     "CIFAR10": load_CIFAR10_data,
#     "CIFAR100": load_CIFAR100_data,
# }


# X_train_public, y_train_public, X_test_public, y_test_public \
# = data_load_functions[public_dataset_name](standarized = True, verbose = True)
# public_dataset = {"X": X_train_public, "y": y_train_public}


# X_train_private, y_train_private, X_test_private, y_test_private \
# = data_load_functions[private_dataset_name](standarized = True, verbose = True)
# private_classes = np.unique(y_train_private)
# private_data, total_private_data\
# =generate_bal_private_data(X_train_private, y_train_private,
#                             N_parties = N_parties, 
#                             classes_in_use = private_classes, 
#                             N_samples_per_class = N_samples_per_class,
#                             data_overlap = False)
# for i in range(N_parties):
#     private_data[i]['y'] = tf.keras.utils.to_categorical(private_data[i]['y'], len(private_classes))
# y_test_private = tf.keras.utils.to_categorical(y_test_private, len(private_classes))
# private_test_data = {"X": X_test_private, "y": y_test_private}

# ============++++++++++++++++++++++++++++++++==================++++++++++++++
# ============++++++++++++++++++++++++++++++++==================++++++++++++++
# ============++++++++++++++++++++++++++++++++==================++++++++++++++


# my_data_dir = '../data'
# if not os.path.exists(my_data_dir):
#     os.mkdir(my_data_dir)

# # create a directory for the private dataset
# dataset_dir = os.path.join(my_data_dir, private_dataset_name)
# if not os.path.exists(dataset_dir):
#     os.mkdir(dataset_dir)

# # create a directory for each client and save its private data
# for i, d in enumerate(private_data) :
#     client_dir = os.path.join(dataset_dir, str(i))
#     if not os.path.exists(client_dir):
#         os.mkdir(client_dir)
#     np.save(os.path.join(client_dir, 'X'), d['X'])
#     np.save(os.path.join(client_dir, 'y'), d['y'])

# # create a directory for the public dataset
# alignment_dir = os.path.join(dataset_dir, 'alignment')
# if not os.path.exists(alignment_dir):
#     os.mkdir(alignment_dir)

# # save the public data in files, one for each class.
# # The data is saved in class-separated files, so the labels are the file names.
# public_classes = np.unique(y_train_public)
# for c in public_classes:
#     X_c = X_train_public[np.where(y_train_public == c)[0]]
#     np.save(os.path.join(alignment_dir, 'X_align_{}'.format(c)), X_c)

# # make a directory for the test data
# # the test data is saved in class-separated files, so the labels are the file names.
# test_dir = os.path.join(my_data_dir, private_dataset_name, 'test')
# if not os.path.exists(test_dir):
#     os.mkdir(test_dir)
# for c in private_classes:
#     X_test_private_c = X_test_private[np.where(y_test_private == c)[0]]
#     np.save(os.path.join(test_dir, 'X_{}'.format(c)), X_test_private_c)
