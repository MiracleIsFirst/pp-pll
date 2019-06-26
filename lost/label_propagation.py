import pandas as pd
import numpy as np
import operator
from functools import reduce
from cvxopt import solvers, matrix


label_count = 16
feature_count = 108
k=10

def my_knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.square(diffMat)
    sqDistances = np.sum(sqDiffMat,axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = np.argsort(distances)

    return sortedDistIndicies[1:k + 1]

if __name__ == "__main__":
    print('*******************闵子剑的论文*******************')
    data = pd.read_csv('lost.txt', header=None)
    header_num = len(data.columns)+1

    data.columns = range(len(data.columns))
    data['id'] = data.index
    data = pd.concat([data[['id']],data.iloc[:,:-1]],axis=1)
    feature = list(range(feature_count))
    new_feature = ['feature'+str(x) for x in feature]
    for i in range(len(feature)):
        data.rename(columns={feature[i]: new_feature[i]}, inplace=True)

    data['TL'] = list(map(lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16: \
                              [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16].index(
                                  max([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16])), \
                          data[header_num - 17], data[header_num - 16], data[header_num - 15], data[header_num - 14],
                          data[header_num - 13], data[header_num - 12], data[header_num - 11], data[header_num - 10],
                          data[header_num - 9], data[header_num - 8], data[header_num - 7], data[header_num - 6],
                          data[header_num - 5], data[header_num - 4], data[header_num - 3], data[header_num - 2]))

    data['PL'] = list(map(lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16: \
                              list((np.where(
                                  np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16])))[
                                       0]), \
                          data[header_num - 17 - 16], data[header_num - 16 - 16], data[header_num - 15 - 16],
                          data[header_num - 14 - 16], data[header_num - 13 - 16], data[header_num - 12 - 16],
                          data[header_num - 11 - 16], data[header_num - 10 - 16], data[header_num - 9 - 16],
                          data[header_num - 8 - 16], data[header_num - 7 - 16], data[header_num - 6 - 16],
                          data[header_num - 5 - 16], data[header_num - 4 - 16], data[header_num - 3 - 16],
                          data[header_num - 2 - 16]))
    feature = new_feature

    for f in feature:
        max = np.max(data[f])
        min = np.min(data[f])

        data[f] = data[f].apply(lambda x: (x - min) / (max - min))

    ######
    #标签变特征
    # parameters = pd.read_csv('em_parameter/parameters_190.csv')
    # print(parameters.shape)
    #
    # data_matrix = data[feature].values
    # probability = data_matrix.dot(parameters)
    # probability = np.exp(probability)
    # sum_probability = np.sum(probability, axis=1).reshape(-1, 1)
    # probability = probability / sum_probability
    #
    # probability = pd.DataFrame(probability)
    # for index, row in data.iterrows():
    #     absense = [x for x in range(label_count) if x not in row['PL']]
    #     probability.loc[index, absense] = 0
    #
    # for f in probability.columns:
    #     probability.rename(columns={f: 'pro' + str(f)}, inplace=True)
    #
    # data = pd.concat([data, probability], axis=1)
    #
    # feature = feature + ['pro' + str(x) for x in range(label_count)]

    ######

    W = [[0.0 for i in range(len(data))] for j in range(len(data))]
    Pro = [[0.0 for i in range(label_count)] for j in range(len(data))]
    Pro = np.array(Pro)
    W = np.array(W)
    for index, row in data.iterrows():
        Pro[index,row['PL']] = 1 / len(row['PL'])

        object_self = data[feature].values[index]
        row_knn_list = my_knn(object_self, data[feature].values, k)
        knn_data = data[feature].values[row_knn_list]
        object_self = object_self.reshape(-1,1)


        knn_data_det = np.linalg.det(knn_data.dot(knn_data.T))
        if(knn_data_det != 0):
            w_index = np.linalg.solve(knn_data.dot(knn_data.T),
                                      knn_data.dot(object_self.reshape(-1, 1)))
        else:
            w_index = np.linalg.solve(knn_data.dot(knn_data.T) + np.eye(10),
                                      knn_data.dot(object_self.reshape(-1, 1)))

        if((False in (w_index>=0)) == True):
            P = 2 * knn_data.dot(knn_data.T)
            q = (-2 * object_self.T.dot(knn_data.T)).reshape(-1, 1)
            G = -1 * np.eye(k)
            h = np.zeros((k, 1))
            w_index = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            W[index, row_knn_list] = np.array(w_index['x']).reshape(1, -1)[0]
            w1 = np.array(w_index['x']).reshape(-1, 1)
            result = (object_self.T - w1.T.dot(knn_data)).dot(object_self - knn_data.T.dot(w1))
        else:
            W[index, row_knn_list] = np.array(w_index).reshape(1, -1)[0]
            w1 = np.array(w_index).reshape(-1, 1)
            result = (object_self.T - w1.T.dot(knn_data)).dot(object_self - knn_data.T.dot(w1))

    D = np.diag(np.sum(W,axis=1))
    H = W.dot(np.linalg.inv(D))

    ###
    #em传播

    parameters = pd.read_csv('em_parameter/parameters_190.csv')
    feature = ['feature' + str(x) for x in range(feature_count)]
    data_matrix = data[feature].values
    probability = data_matrix.dot(parameters)
    print(probability.shape)

    probability = np.exp(probability)
    sum_probability = np.sum(probability, axis=1).reshape(-1, 1)
    probability = probability / sum_probability

    probability = pd.DataFrame(probability)
    for index, row in data.iterrows():
        absense = [x for x in range(label_count) if x not in row['PL']]
        probability.loc[index, absense] = 0
    Pro = probability.values

    ###

    F_t = Pro
    alpha = 0.55
    for t in range(100):
        F_t_plus_1 = alpha * H.dot(F_t) + (1-alpha) * Pro
        for index, row in data.iterrows():
            absense = [x for x in range(label_count) if x not in row['PL']]
            F_t_plus_1[index, absense] = 0
            F_t_plus_1[index, row['PL']] = F_t_plus_1[index, row['PL']] / np.sum(F_t_plus_1[index, row['PL']])

        if(np.sum(abs(F_t_plus_1-F_t)) <= 10e-3):
            F_t = F_t_plus_1
            break
        F_t = F_t_plus_1


    data['pre_label'] = np.argmax(F_t, axis=1)
    accu = list(map(lambda x, y: 1 if x == y else 0, data['TL'], data['pre_label']))
    print('accuracy:', np.sum(accu) / len(accu))

    F_t = pd.DataFrame(F_t)

    # F_t.to_csv('prior_probability2.csv',index = None, header=None)

