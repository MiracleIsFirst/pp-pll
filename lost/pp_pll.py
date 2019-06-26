import pandas as pd
import numpy as np
from cvxopt import solvers, matrix
from sklearn.cluster import KMeans
import math
import time
import matplotlib.pyplot as plt

label_count = 16
feature_count = 108
k=10
my_lambda = 0.005
alpha = 1 / (1 + 1)
E = []
iter_run = [5, 20, 40, 60, 80, 100]

def my_knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.square(diffMat)
    sqDistances = np.sum(sqDiffMat,axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = np.argsort(distances)

    return sortedDistIndicies[1:k + 1]

def label_propagation(data, Pro):
    W = [[0.0 for i in range(len(data))] for j in range(len(data))]
    W = np.array(W)
    for index, row in data.iterrows():
        object_self = data[feature].values[index]
        row_knn_list = my_knn(object_self, data[feature].values, k)
        knn_data = data[feature].values[row_knn_list]
        object_self = object_self.reshape(-1, 1)

        knn_data_det = np.linalg.det(knn_data.dot(knn_data.T))
        if (knn_data_det != 0):
            w_index = np.linalg.solve(knn_data.dot(knn_data.T),
                                      knn_data.dot(object_self.reshape(-1, 1)))
        else:
            w_index = np.linalg.solve(knn_data.dot(knn_data.T) + np.eye(10),
                                      knn_data.dot(object_self.reshape(-1, 1)))

        if ((False in (w_index >= 0)) == True):
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

    D = np.diag(np.sum(W, axis=1))
    H = W.dot(np.linalg.inv(D))

    F_t = Pro
    for t in range(100):
        F_t_plus_1 = alpha * H.dot(F_t) + (1 - alpha) * Pro
        for index, row in data.iterrows():
            absense = [x for x in range(label_count) if x not in row['PL']]
            F_t_plus_1[index, absense] = 0
            F_t_plus_1[index, row['PL']] = F_t_plus_1[index, row['PL']] / np.sum(F_t_plus_1[index, row['PL']])

        if (np.sum(abs(F_t_plus_1 - F_t)) <= 10e-3):
            F_t = F_t_plus_1
            break
        F_t = F_t_plus_1

    return F_t

#(1,1)
def func(parameters, data, probability_e_step):
    feature = ['feature' + str(x) for x in range(feature_count)]
    logistic_matrix = data[feature].values.dot(parameters.T)#(4998,13)
    M = np.max(logistic_matrix, axis=1)

    probability_m_step_second_term = np.sum(np.exp(logistic_matrix - M.reshape(-1,1)), axis=1).reshape(-1,1)
    probability_m_step_second_term = np.log(probability_m_step_second_term)
    probability_m_step = logistic_matrix - M.reshape(-1, 1) - probability_m_step_second_term

    first_term = probability_m_step * probability_e_step #(4998,13)

    second_term = np.square(parameters)
    second_term = (my_lambda / 2) * np.sum(second_term)

    return second_term - np.sum(first_term)

#(13,38)
def gfunc(parameters, data, probability_e_step):
    feature = ['feature' + str(x) for x in range(feature_count)]

    logistic_matrix = data[feature].values.dot(parameters.T)#(4998,13)
    M = np.max(logistic_matrix, axis = 1)
    sum = np.sum(np.exp(logistic_matrix - M.reshape(-1,1)), axis=1).reshape(-1,1)

    probability_m_step = np.exp(logistic_matrix - M.reshape(-1,1)) / sum

    probability = probability_m_step - probability_e_step
    first_term = probability.T.dot(data[feature].values)

    second_term = my_lambda * parameters

    return first_term + second_term

# BFGS
def BFGS(data, parameters2, probability_e_step):
    print('BFGS...')
    rho = 0.55
    max_bfgs_iter = 2000
    beta_piao = 0.4
    beta = 0.55

    Bk = np.eye(feature_count * label_count)  # (494, 494)
    bfgs_iter = 0
    result = []
    while (bfgs_iter <= max_bfgs_iter):
        fk = func(parameters2, data, probability_e_step)  # (1,1)
        gk = gfunc(parameters2, data, probability_e_step).reshape(-1, 1)  # (494,1)
        # 下降方向
        dk = -np.linalg.solve(Bk, gk)  # (494,1)
        m = 0
        mk = 0
        while (m < 100):
            newfk = func(parameters2 + (rho ** m) * dk.reshape(label_count, feature_count),
                         data, probability_e_step)
            newgk = gfunc(parameters2 + (rho ** m) * dk.reshape(label_count, feature_count),
                          data, probability_e_step).reshape(-1, 1)
            if (newfk <= fk + beta_piao * (rho ** m) * dk.T.dot(gk)[0, 0]):
                if (dk.T.dot(newgk)[0, 0] >= beta * dk.T.dot(gk)[0, 0]):
                    mk = m
                    break
            m += 1
        new_parameters2 = parameters2 + (rho ** mk) * dk.reshape(label_count,feature_count)
        sk = (new_parameters2 - parameters2).reshape(-1, 1)  # (494,1)
        y_left = gfunc(new_parameters2, data, probability_e_step).reshape(-1, 1)
        yk = y_left - gk  # (494,1)
        Bk = Bk - (Bk.dot(sk.dot(sk.T)).dot(Bk)) / (sk.T.dot(Bk.dot(sk))) + (yk.dot(yk.T)) / (yk.T.dot(sk))
        parameters2 = new_parameters2
        bfgs_iter += 1
        if (np.sum(abs(y_left)) < 10):
            break
    a = func(parameters2, data, probability_e_step)
    b = np.sum(abs(y_left))
    print('BFGS have finished, and final func is %f, gfunc is %f'%(a,b))
    return parameters2

def PL_EM(data):
    feature = ['feature' + str(x) for x in range(feature_count)]

    parameters = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
    parameters = np.array(parameters)#(13,38)

    for em_iter in range(100):
        print('EM第%d次迭代' % (em_iter), end=': \n')
        # E_STEP
        print('E-Step')
        logistic_matrix = data[feature].values.dot(parameters.T)  # (4998,13)
        M = np.max(logistic_matrix, axis=1)
        sum = np.sum(np.exp(logistic_matrix - M.reshape(-1, 1)), axis=1).reshape(-1, 1)

        probability_e_step = np.exp(logistic_matrix - M.reshape(-1,1)) / sum#(4998,13)
        probability_e_step = pd.DataFrame(probability_e_step)

        for index, row in data.iterrows():
            absense = [x for x in range(label_count) if x not in row['PL']]
            probability_e_step.loc[index, absense] = 0
            probability_e_step.loc[index, row['PL']] = \
                probability_e_step.loc[index, row['PL']] / \
                np.sum(probability_e_step.iloc[index][row['PL']].values)
        probability_e_step = probability_e_step.values #(4998,13)
        probability_e_step = label_propagation(data, probability_e_step)

        # M_STEP
        print('M-Step')
        parameters2 = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
        parameters2 = np.array(parameters2) #(13,38)

        #BFGS
        parameters2 = BFGS(data, parameters2, probability_e_step)
        e = np.sum(abs(parameters2 - parameters))
        print('e: ', e)
        if(e < 0.001):
            break
        E.append(e)
        parameters = parameters2

        if (em_iter % 10 == 0):
            # parameters_temp = pd.DataFrame(parameters.T)
            # parameters_temp.to_csv('em_parameter/parameters_%d.csv' % (em_iter), index=None)

            data_matrix = data[feature].values
            probability = data_matrix.dot(parameters.T)
            probability = np.exp(probability)
            sum_probability = np.sum(probability, axis=1).reshape(-1, 1)
            probability = probability / sum_probability

            probability = pd.DataFrame(probability)
            for index, row in data.iterrows():
                absense = [x for x in range(label_count) if x not in row['PL']]
                probability.loc[index, absense] = 0
            probability = probability.values

            data['pre_label'] = np.argmax(probability, axis=1)
            accu = list(map(lambda x, y: 1 if x == y else 0, data['TL'], data['pre_label']))
            print('accuracy_%d:' % (em_iter), np.sum(accu) / len(accu))

            data.drop(['pre_label'], axis=1, inplace=True)
    return parameters

if __name__ == '__main__':
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

    train = data.sample(frac=0.5).reset_index(drop=True)
    test = data[~data.id.isin(list(set(train['id'])))].reset_index(drop=True)

    parameters = PL_EM(train)

    data_matrix = test[feature].values
    probability = data_matrix.dot(parameters.T)
    probability = np.exp(probability)
    sum_probability = np.sum(probability, axis=1).reshape(-1, 1)
    probability = probability / sum_probability

    probability = pd.DataFrame(probability)
    for index, row in test.iterrows():
        absense = [x for x in range(label_count) if x not in row['PL']]
        probability.loc[index, absense] = 0
    probability = probability.values

    test['pre_label'] = np.argmax(probability, axis=1)
    accu = list(map(lambda x, y: 1 if x == y else 0, test['TL'], test['pre_label']))
    print('accuracy:', np.sum(accu) / len(accu))
    print(E)

