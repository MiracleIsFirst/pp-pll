import pandas as pd
import numpy as np
from cvxopt import solvers, matrix

label_count = 219
feature_count = 163
k=10
my_lambda = 0.005
alpha = 1 / (1 + 1)
T = 100

def my_knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.square(diffMat)
    sqDistances = np.sum(sqDiffMat,axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = np.argsort(distances)

    return sortedDistIndicies[1:k + 1]

def stage_one(data):
    W = [[0.0 for i in range(len(data))] for j in range(len(data))]
    W = np.array(W)
    for index, row in data.iterrows():
        # find k-nearest neighbor
        object_self = data[feature].values[index]
        row_knn_list = my_knn(object_self, data[feature].values, k)
        knn_data = data[feature].values[row_knn_list]  # (k, feature_count)
        object_self = object_self.reshape(-1, 1)  # (feature_count, 1)
        knn_data_det = np.linalg.det(knn_data.dot(knn_data.T))  # calculate matrix determinant

        if (knn_data_det != 0):
            # solve a linear matrix equation
            w_index = np.linalg.solve(knn_data.dot(knn_data.T),
                                      knn_data.dot(object_self))
        else:
            w_index = np.zeros((k, 1))

        if ((False in (w_index >= 0)) | (np.sum(w_index) == 0)):
            P = 2 * knn_data.dot(knn_data.T)
            q = (-2 * object_self.T.dot(knn_data.T)).reshape(-1, 1)
            G = -1 * np.eye(k)
            h = np.zeros((k, 1))
            w_index = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            W[index, row_knn_list] = np.array(w_index['x']).reshape(1, -1)[0]
        else:
            W[index, row_knn_list] = np.array(w_index).reshape(1, -1)[0]

    # normalization
    D = np.diag(np.sum(W, axis=1))
    H = W.dot(np.linalg.inv(D))

    return H

def probability_propagation(data, Pro, H):
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

#(label_count,feature_count)
def func(parameters, data, probability_e_step):
    feature = ['feature' + str(x) for x in range(feature_count)]
    logistic_matrix = data[feature].values.dot(parameters.T)
    M = np.max(logistic_matrix, axis=1)

    probability_m_step_second_term = np.sum(np.exp(logistic_matrix - M.reshape(-1,1)), axis=1).reshape(-1,1)
    probability_m_step_second_term = np.log(probability_m_step_second_term)
    probability_m_step = logistic_matrix - M.reshape(-1, 1) - probability_m_step_second_term

    first_term = probability_m_step * probability_e_step

    second_term = np.square(parameters)
    second_term = (my_lambda / 2) * np.sum(second_term)

    return second_term - np.sum(first_term)

#(label_count,feature_count)
def gfunc(parameters, data, probability_e_step):
    feature = ['feature' + str(x) for x in range(feature_count)]

    logistic_matrix = data[feature].values.dot(parameters.T)
    M = np.max(logistic_matrix, axis = 1)
    sum = np.sum(np.exp(logistic_matrix - M.reshape(-1,1)), axis=1).reshape(-1,1)

    probability_m_step = np.exp(logistic_matrix - M.reshape(-1,1)) / sum

    probability = probability_m_step - probability_e_step
    first_term = probability.T.dot(data[feature].values)

    second_term = my_lambda * parameters

    return first_term + second_term

# BFGS
def L_BFGS(data, parameters2, probability_e_step):
    print('L_BFGS...')
    rho = 0.55
    max_bfgs_iter = 2000
    beta_piao = 0.4
    beta = 0.55

    Bk = np.eye(feature_count * label_count)
    gk = gfunc(parameters2, data, probability_e_step).reshape(-1, 1)
    dk = -np.linalg.solve(Bk, gk)
    bfgs_iter = 0

    s = []
    y = []
    l = 7

    while (bfgs_iter <= max_bfgs_iter):
        fk = func(parameters2, data, probability_e_step)  # (1,1)
        gk = gfunc(parameters2, data, probability_e_step).reshape(-1, 1)  # (494,1)
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
        new_parameters2 = parameters2 + (rho ** mk) * dk.reshape(label_count, feature_count)

        if (bfgs_iter > l):
            s.pop(0)
            y.pop(0)

        sk = (new_parameters2 - parameters2).reshape(-1, 1)
        y_left = gfunc(new_parameters2, data, probability_e_step).reshape(-1, 1)
        yk = y_left - gk

        s.append(sk)
        y.append(yk)

        t = len(s)
        a = []
        for i in range(t):
            alpha = s[t - i - 1].T.dot(y_left) / (y[t - i - 1].T.dot(s[t - i - 1]))
            y_left = y_left - alpha[0,0]* y[t - i - 1]
            a.append(alpha[0,0])

        r = Bk.dot(y_left)

        for i in range(t):
            gama = y[i].T.dot(r) / (y[i].T.dot(s[i]))
            r = r + s[i]*(a[t - i - 1] - gama[0,0])

        if(yk.T.dot(sk) > 0):
            dk = -r

        parameters2 = new_parameters2
        bfgs_iter += 1
        if (np.sum(abs(y_left)) < 2000):
            break
    a = func(parameters2, data, probability_e_step)
    b = np.sum(abs(y_left))
    print('L_BFGS have ended, and the value of final func is %f, the derivative of final func is %f' % (a, b))
    return parameters2

def runPP_PLL(data):
    print('Start PP-PLL')
    feature = ['feature' + str(x) for x in range(feature_count)]

    parameters = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
    parameters = np.array(parameters)#(label_count,feature_count)
    print('Stage 1: ')
    W = stage_one(data)

    print()
    print('Stage 2:')
    for em_iter in range(T):
        print('The %d-th iteration of PP-PLL' % (em_iter), end=': \n')

        # E_STEP
        print('E-Step')
        logistic_matrix = data[feature].values.dot(parameters.T)  #(instance_length,label_count)
        M = np.max(logistic_matrix, axis=1)
        sum = np.sum(np.exp(logistic_matrix - M.reshape(-1, 1)), axis=1).reshape(-1, 1)

        probability_e_step = np.exp(logistic_matrix - M.reshape(-1, 1)) / sum  #(instance_length,label_count)

        for index, row in data.iterrows():
            absense = [x for x in range(label_count) if x not in row['PL']]
            sum = np.sum(probability_e_step[index, row['PL']])

            for i in range(label_count):
                if (i in absense):
                    probability_e_step[index, i] = 0
                else:
                    probability_e_step[index, i] = probability_e_step[index, i] / sum
        probability_e_step = probability_propagation(data, probability_e_step, W) #(instance_length,label_count)

        # M_STEP
        print('M-Step')
        parameters2 = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
        parameters2 = np.array(parameters2) #(label_count,feature_count)

        #BFGS
        parameters2 = L_BFGS(data, parameters2, probability_e_step)
        e = np.sum(abs(parameters2 - parameters))
        print('e: ', e)
        if(e < 1500):
            break
        parameters = parameters2
    return parameters

if __name__ == '__main__':
    data = pd.read_csv('YahooNews.txt')

    print('data processing...')
    header_num = len(data.columns)+1
    data.columns = range(len(data.columns))
    data['id'] = data.index
    data = pd.concat([data[['id']],data.iloc[:,:-1]],axis=1)
    feature = list(range(feature_count))
    new_feature = ['feature'+str(x) for x in feature]
    for i in range(len(feature)):
        data.rename(columns={feature[i]: new_feature[i]}, inplace=True)

    feature = ['feature'+str(x) for x in range(feature_count)]
    label = [x for x in data.columns if x not in feature + ['id']]
    PL = label[:label_count]
    TL = label[label_count:]

    PL_list = []
    TL_list = []
    for index, row in data.iterrows():
        PL_list.append(list(np.where(row[PL] == 1)[0]))
        TL_list.append(list(np.where(row[TL] == 1)[0])[0])

    data['PL'] = PL_list
    data['TL'] = TL_list

    for f in feature:
        max = np.max(data[f])
        min = np.min(data[f])

        data[f] = data[f].apply(lambda x: (x - min) / (max - min))

    print('data processing have ended!')

    parameters = runPP_PLL(data)

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
    print('accuracy:', np.sum(accu) / len(accu))



