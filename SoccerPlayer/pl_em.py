import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math
import time
import matplotlib.pyplot as plt

label_count = 171
feature_count = 279

#(1,1)
def func(parameters, data, probability_e_step):
    my_lambda = 1e-4
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
    my_lambda = 1e-4
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
def L_BFGS(data, parameters2, probability_e_step):
    print('L_BFGS...')
    rho = 0.55
    max_bfgs_iter = 2000
    beta_piao = 0.4
    beta = 0.55

    Bk = np.eye(feature_count * label_count)  # (494, 494)
    gk = gfunc(parameters2, data, probability_e_step).reshape(-1, 1)  # (494,1)
    # 下降方向
    dk = -np.linalg.solve(Bk, gk)  # (494,1)
    bfgs_iter = 0
    result = []

    # s1和s2用于保存最近l个,这里l取6
    s = []
    y = []
    l = 6

    while (bfgs_iter <= max_bfgs_iter):
        print('BFGS第%d次迭代' % (bfgs_iter), end=': \n')
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

        if(mk == m):
            print('寻找步长结束，步长为0.55的%d次方' % (mk))
        else:
            print('寻找步长结束，未找到合适的步长')
        new_parameters2 = parameters2 + (rho ** mk) * dk.reshape(label_count, feature_count)

        # 保留l个
        if (bfgs_iter > l):
            s.pop(0)
            y.pop(0)

        # 计算最新的
        sk = (new_parameters2 - parameters2).reshape(-1, 1)  # (494,1)
        y_left = gfunc(new_parameters2, data, probability_e_step).reshape(-1, 1)
        yk = y_left - gk  # (494,1)

        s.append(sk)
        y.append(yk)

        # two-loop的过程
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

        print('parameters: ', parameters2.reshape(-1, 1).tolist())
        result.append(func(parameters2, data, probability_e_step))
        print('func: ', result[-5:])
        print('gfunc: ', np.sum(abs(y_left)))

        parameters2 = new_parameters2
        bfgs_iter += 1
        if (np.sum(abs(y_left)) < 100):
            break
    a = func(parameters2, data, probability_e_step)
    b = np.sum(abs(y_left))
    print('BFGS have finished, and final func is %f, gfunc is %f' % (a, b))
    return parameters2

def PL_EM(data):
    feature = ['feature' + str(x) for x in range(feature_count)]

    parameters = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
    parameters = np.array(parameters)#(13,38)

    for em_iter in range(5):
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

        # M_STEP
        print('M-Step')
        parameters2 = [[1 / label_count for x in range(feature_count)] for y in range(label_count)]
        parameters2 = np.array(parameters2) #(13,38)

        #BFGS
        parameters2 = L_BFGS(data, parameters2, probability_e_step)
        e = np.sum(abs(parameters2 - parameters))
        print('e: ', e)
        if(e < 10):
            break
        parameters = parameters2

        if (em_iter % 10 == 0):
            parameters_temp = pd.DataFrame(parameters.T)
            parameters_temp.to_csv('em_parameter/parameters_%d.csv' % (em_iter), index=None)

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
    data = pd.read_csv('SoccerPlayer2.txt')
    data = data.head(1000)
    feature = ['feature'+str(x) for x in range(feature_count)]
    label = [x for x in data.columns if x not in feature + ['id']]
    PL = label[:label_count]
    TL = label[label_count:]
    print('data processing...')
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

    print('data processing have finished!')

    parameters = PL_EM(data)

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



