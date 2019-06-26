import pandas as pd
import numpy as np
import random

feature_count = 7
class_count = 8
p = 0.4

if __name__ == '__main__':
    data = pd.read_csv(r'ecoli.csv', header = None)

    data_matrix = []
    label_list = []
    for index, row in data.iterrows():
        temp = []
        for st in row.values[0].split(' '):
            if(st != ''):
                if('.' in st):
                    temp.append(float(st))
                else:
                    if(len(st) <= 3):
                        label_list.append(st)


        data_matrix.append(temp)

    data_matrix = np.array(data_matrix)
    data = pd.DataFrame(data_matrix)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['label'] = le.fit_transform(label_list)

    for i in range(feature_count):
        data.rename(columns = {i : 'feature'+str(i)}, inplace = True)

    pl_random_index = []
    r_random_index = []
    for i in range(int((len(data))*p)):
        pl_index = random.randrange(0, len(data), 1)
        r = random.randrange(1, 4, 1)
        r_random_index.append(r)
        while(pl_index in pl_random_index):
            pl_index = random.randrange(0, len(data), 1)
        pl_random_index.append(pl_index)

    PL_list = []
    r_index = 0
    for index, row in data.iterrows():
        temp = []
        temp.append(int(row['label']))
        if(index in pl_random_index):
            for t in range(r_random_index[r_index]):
                pl = random.randrange(0, class_count, 1)
                while (pl in temp):
                    pl = random.randrange(0, class_count, 1)
                temp.append(pl)
            r_index += 1
        temp.sort()
        PL_list.append(temp)

    data['PL'] = PL_list


    data.to_csv('pl_ecoli.csv', index = None)