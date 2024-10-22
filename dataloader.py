import numpy as np
import json

def load_dataset():
    f = open("dataset_info.json")
    dataset_info = json.load(f)
    f.close()
    dataset_info['train_data'] = np.load(dataset_info['train_data_path'])
    dataset_info['val_data'] = np.load(dataset_info['val_data_path'])
    dataset_info['test_data'] = np.load(dataset_info['test_data_path'])
    return dataset_info


def load_citeulike():

    raw_data = dict()
    raw_data['total_users'] = 5551
    raw_data['total_items'] = 16980
    
    raw_data['train_data'] = np.load('dataset/citeulike/rsrf_user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/citeulike/rsrf_user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/citeulike/rsrf_user_data_test.npy')
    
    return raw_data

def load_lastfm():

    raw_data = dict()
    raw_data['total_users'] = 992
    raw_data['total_items'] = 14598

    raw_data['train_data'] = np.load('dataset/lastfm/rsrf_user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/lastfm/rsrf_user_data_test.npy')
    raw_data['test_data'] = np.load('dataset/lastfm/rsrf_user_data_test.npy')
    
    return raw_data

def load_tradesy():

    raw_data = dict()
    raw_data['total_users'] = 19243
    raw_data['total_items'] = 165906
    
    raw_data['train_data'] = np.load('dataset/tradesy/rsrf_user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/tradesy/rsrf_user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/tradesy/rsrf_user_data_test.npy')

    return raw_data

def load_kkbox():

    raw_data = dict()

    raw_data['train_data'] = np.load("dataset/kkbox/traindata.npy")
    raw_data['val_data'] = np.load("dataset/kkbox/valdata.npy")
    raw_data['test_data'] = np.load("dataset/kkbox/testdata.npy")

    raw_data['total_users'] = 27113
    raw_data['total_items'] = 223723
    return raw_data

