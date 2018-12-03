import numpy as np

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

def load_taobao():

    raw_data = dict()

    raw_data['train_data'] = np.load("dataset/taobao/traindata.npy")
    raw_data['val_data'] = np.load("dataset/taobao/valdata.npy")
    raw_data['test_data'] = np.load("dataset/taobao/testdata.npy")

    raw_data['total_users'] = 20000
    raw_data['total_items'] = 643380
    return raw_data

def load_amazon_book():

    raw_data = dict()
    raw_data['max_user'] = 99473
    raw_data['max_item'] = 450166

    raw_data['train_data'] = np.load('dataset/amazon/user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/amazon/user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/amazon/user_data_test.npy')

    raw_data['item_features'] = np.array(np.memmap('dataset/amazon/book_features_update.mem', 
                                dtype=np.float32, mode='r', shape=(raw_data['max_item'], 4096)))
    raw_data['user_features'] = np.load('dataset/amazon/user_features_categories.npy')
    return raw_data
