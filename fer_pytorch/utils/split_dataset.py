import  pandas as pd

'''
Split dataframe to n pieces, each piece with train, val
@param df
@param n:  n pieces
@param train_ratio:  train ratio in each piece
@param seed: random seed for shuffle
'''
def split_to_n_pieces_with_train_val(
        df: pd.DataFrame,
        n = 1,
        train_ratio = 0.7,
        seed = 666):
    ## ramdom shuffle
    df = df.sample(frac=1, random_state= seed)
    samples_list = []
    N = len(df)//n
    for i in range(n -1):
        samples_list.append(df[i*N : (i+1) * N])
    samples_list.append(df[(n - 1)*N : ])

    pieces = []
    for i, d in enumerate(samples_list):
        train_n  = int(train_ratio * len(d))
        train_df = d[:train_n]
        val_df   = d[train_n: ]
        pieces.append((train_df, val_df))
    return  pieces