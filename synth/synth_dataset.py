import numpy as np
import matplotlib.pyplot as plt


dataset = np.genfromtxt('resources/imdb.csv', delimiter=',', dtype='str')

folders = dataset[:, 13]

## Remove folders 1, 2
idx = np.flatnonzero((folders != '1') & (folders != '2'))
dataset = dataset[idx]

dataset = np.random.permutation(dataset)

third = int(len(dataset) / 3)

db1 = dataset[:third]
db2 = dataset[third:2 * third]
db3 = dataset[2 * third:]

np.savetxt('resources/imdb_synth/imdb_1.csv', db1, fmt='%s', delimiter=',')
np.savetxt('resources/imdb_synth/imdb_2.csv', db2, fmt='%s', delimiter=',')
np.savetxt('resources/imdb_synth/imdb_3.csv', db3, fmt='%s', delimiter=',')



dataset = db1

all_folders = dataset[:, 13].astype(int)

ages = dataset[:, 10].astype(int)

fig, ax = plt.subplots(3, 1)

folders_8 = [
    [[3, 4, 5, 6, 7, 8], [9], [10]],
    [[5, 6, 7, 8, 9, 10], [3], [4]],
    [[3, 4, 5, 8, 9, 10], [6], [7]]
]

for i, split in enumerate(folders_8):

    for folders, label in zip(split, ['trn', 'val', 'tst']):

        mask = np.full(len(all_folders), 0, dtype=bool)

        for folder in folders:
            mask = np.bitwise_or(mask, all_folders == folder)
        
        a = ages[mask]

        ages_dict = {age: 0 for age in range(1, 91)}
        
        for a_i in a:
            ages_dict[a_i] += 1

        keys = list(ages_dict.keys())

        values = list(ages_dict.values())
        values = np.array(values) / np.sum(values)

        ax[i].plot(keys, values, label=label)

        ax[i].legend()

plt.show()