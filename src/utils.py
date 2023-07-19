import networkx as nx
import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import logging

def collate_data(data):
    
    subgraph_list1, subgraph_list2, subgraph_list3, label_list = map(list, zip(*data))
    subgraph1 = dgl.batch(subgraph_list1)
    subgraph2 = dgl.batch(subgraph_list2)
    subgraph3 = dgl.batch(subgraph_list3)
    g_label = torch.stack(label_list)
    
    return subgraph1, subgraph2, subgraph3, g_label
    
def get_logger(name, path):
    
    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)
    
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

def kddcup_load():
    path = 'data/kddcup15'

    course_date = pd.read_csv(os.path.join(path, 'date.csv'))
    course_info = pd.read_csv(os.path.join(path, 'object.csv'))
    enrollment_train = pd.read_csv(os.path.join(path, 'enrollment_train.csv'))
    enrollment_test = pd.read_csv(os.path.join(path, 'enrollment_test.csv'))
    log_train = pd.read_csv(os.path.join(path, 'log_train.csv'))
    log_test = pd.read_csv(os.path.join(path, 'log_test.csv'))
    truth_train = pd.read_csv(os.path.join(path, 'truth_train.csv'), header=None, names=['enrollment_id', 'truth'])
    truth_test = pd.read_csv(os.path.join(path, 'truth_test.csv'), header=None, names=['enrollment_id', 'truth'])
    
    truth_all = pd.concat([truth_train, truth_test])
    log_all = pd.concat([log_train, log_test])
    enrollment_all = pd.concat([enrollment_train, enrollment_test])

    return course_date, course_info, enrollment_train, enrollment_test, log_train, log_test, truth_train, truth_test, truth_all, log_all, enrollment_all