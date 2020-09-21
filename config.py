
#!------------
# Model params
#!------------


epochs = 100
lr = 0.0005


#!------------
# Define folders
# Each row is a new fold
# Each column [trn], [val], [tst] has a list of folders
#!------------


folders_2 = [
    [[1,2], [3,5], [4]]
]

folders_3 = [
    [[1], [2], [3]],
    [[2], [3], [1]],
    [[3], [1], [2]]
]

folders_8 = [
    [[3, 4, 5, 6, 7, 8], [9], [10]],
    [[5, 6, 7, 8, 9, 10], [3], [4]],
    [[3, 4, 5, 8, 9, 10], [6], [7]]
]

folders_10 = [
    [[1, 2, 3, 4, 5, 6, 7], [8], [9, 10]],
    [[1, 6, 2, 9, 3, 10, 8], [4], [5, 7]],
    [[7, 9, 10, 5, 8, 3, 4], [6], [2, 1]]
]


## use tuple (path, folders) to add datasets
datasets = {
    # 0: ('resources/agedb.csv', folders_10),
    # 1: ('resources/morph.csv', folders_10),
    # 2: ('resources/appa_real.csv', folders_3),
    # 3: ('resources/imdb.csv', folders_10),
    # 4: ('resources/utkf.csv', folders_3),
    # 5: ('resources/lfw.csv', folders_10),
    # 6: ('resources/cpmrd.csv', folders_10),
    # 7: ('resources/inet.csv', folders_10),
    # 8: ('resources/group_photos.csv', folders_10),
    # 9: ('resources/pal.csv', folders_10),
    # 10: ('resources/pub_fig.csv', folders_10),
    # 11: ('resources/school_classes.csv', folders_10),

    ## synthetic IMDB
    0: ('resources/imdb.csv', folders_2),

    # 0: ('resources/imdb_synth/imdb_synth_1.csv', folders_8),
    # 1: ('resources/imdb_synth/imdb_synth_2.csv', folders_8),
    # 2: ('resources/imdb_synth/imdb_synth_3.csv', folders_8),

    # 0: ('resources/imdb_synth/imdb_synth_1.csv', folders_8),
    # 1: ('resources/imdb_synth/imdb_synth_2.csv', folders_8),
    # 2: ('resources/imdb_synth/imdb_synth_3.csv', folders_8),

    ## synthetic morph
    # 0: ('resources/morph_synth/morph_synth_part1.csv', folders_10),
    # 1: ('resources/morph_synth/morph_synth_part2.csv', folders_10),
    # 2: ('resources/morph_synth/morph_synth_part3.csv', folders_10)
}