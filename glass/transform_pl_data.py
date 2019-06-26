import pandas as pd
import numpy as np
import random

p = 0.7
if __name__ == '__main__':
    data = pd.read_csv(r'glass.csv', header = None,
                       names=['instance_id']
                             +['feature'+str(i) for i in range(9)]
                             +['label'])

    data.loc[data.label == 1, 'label'] = 0
    data.loc[data.label == 2, 'label'] = 1
    data.loc[data.label == 3, 'label'] = 2
    data.loc[data.label == 5, 'label'] = 3
    data.loc[data.label == 6, 'label'] = 4
    data.loc[data.label == 7, 'label'] = 5

    pl_random_index = []
    r_random_index = []
    for i in range(int(214*p)):
        pl_index = random.randrange(0, 214, 1)
        r = random.randrange(1, 4, 1)
        r_random_index.append(r)
        while(pl_index in pl_random_index):
            pl_index = random.randrange(0, 214, 1)
        pl_random_index.append(pl_index)

    PL_list = []
    r_index = 0
    for index, row in data.iterrows():
        temp = []
        temp.append(int(row['label']))
        if(index in pl_random_index):
            for t in range(r_random_index[r_index]):
                pl = random.randrange(0, 6, 1)
                while (pl in temp):
                    pl = random.randrange(0, 6, 1)
                temp.append(pl)
            r_index += 1
        temp.sort()
        PL_list.append(temp)

    data['PL'] = PL_list


    data.to_csv('pl_glass.csv', index = None)