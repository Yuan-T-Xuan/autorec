import csv
import numpy as np

with open('traindata.csv') as csvfile:
    traindata = list(csv.reader(csvfile))
    train_structured_arr = np.zeros(len(traindata)-1, 
                                    dtype=[('user_id', np.int32), ('item_id', np.int32)])
    for i, row in enumerate(traindata[1:]):
        train_structured_arr[i] = (int(row[0]), int(row[1]))
    np.save('traindata.npy', train_structured_arr)

with open('valdata.csv') as csvfile:
    valdata = list(csv.reader(csvfile))
    val_structured_arr = np.zeros(len(valdata)-1, 
                                  dtype=[('user_id', np.int32), ('item_id', np.int32)])
    for i, row in enumerate(valdata[1:]):
        val_structured_arr[i] = (int(row[0]), int(row[1]))
    np.save('valdata.npy', val_structured_arr)

with open('testdata.csv') as csvfile:
    testdata = list(csv.reader(csvfile))
    test_structured_arr = np.zeros(len(testdata)-1, 
                                   dtype=[('user_id', np.int32), ('item_id', np.int32)])
    for i, row in enumerate(testdata[1:]):
        test_structured_arr[i] = (int(row[0]), int(row[1]))
    np.save('testdata.npy', test_structured_arr)

