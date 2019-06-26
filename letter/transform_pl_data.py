import pandas as pd
import numpy as np
import random

feature_count = 16
class_count = 26
p = 0.7

if __name__ == '__main__':
    data = pd.read_csv(r'letter-recognition.csv', header = None)
    data.rename(columns = {0 : 'label'}, inplace = True)

    for i in range(1, len(data.columns)):
        data.rename(columns={i: 'feature'+str(i-1)}, inplace=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['label'] = le.fit_transform(list(data['label']))

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


    data.to_csv('pl_letter.csv', index = None)