# ---------------
## Model params
# ---------------


epochs = 200
lr = 0.001



# ----------------
## Define folders 
# 
# Each row is a new fold
# Each column [trn], [val], [tst] has a list of folders
# 
# ----------------

agedb_folders = [
    [[1, 2, 3, 4, 5, 6, 7], [8], [9, 10]],
    [[1, 6, 2, 9, 3, 10, 8], [4], [5, 7]],
    [[7, 9, 10, 5, 8, 3, 4], [6], [2, 1]]
]

morph_folders = [
    [[1, 2, 3, 4, 5, 6, 7], [8], [9, 10]],
    [[1, 6, 2, 9, 3, 10, 8], [4], [5, 7]],
    [[7, 9, 10, 5, 8, 3, 4], [6], [2, 1]]
]

appa_real_folders = [
    [[1], [2], [3]],
    [[2], [3], [1]],
    [[3], [1], [2]]
]


## use tuple (path, folders) to add datasets
datasets = [
    ('resources/agedb.csv', 'resources/agedb_landmarks.csv', agedb_folders),
    ('resources/morph.csv', 'resources/morph_landmarks.csv', morph_folders),
    ('resources/appa_real.csv', 'resources/appa_real_landmarks.csv', appa_real_folders)
]