import sys
sys.path.append('../')

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocess import add_time_idx

def train_val_test_split(
    clickstream_df,
    test_quantile=0.9,
    test_validation_ratio=0.5,
    filter_negative=False,
    rating_threshold=3.5,
):
    """
    Split clickstream by date.
    
    Split validation and test by users in `test_validation_ratio` proportion.
    """
    clickstream_df = clickstream_df.sort_values(['user_id', 'timestamp'])
    test_timepoint = clickstream_df['timestamp'].quantile(
    q=test_quantile, interpolation='nearest'
    )
    test_full = clickstream_df.query('timestamp >= @test_timepoint')
    train = clickstream_df.drop(test_full.index)
    
    train = add_time_idx(train)
    test_full = add_time_idx(test_full)

    if filter_negative:
        train = train.query('rating >= @rating_threshold')
        
    test_full = test_full[test_full['user_id'].isin(train['user_id'])]
    test_full = test_full[test_full['item_id'].isin(train['item_id'])]
    
    val, test = train_test_split(
        test_full,
        test_size=test_validation_ratio,
        random_state=42
    )

    val_1, val_2 = train_test_split(
        val,
        test_size=test_validation_ratio,
        random_state=42
    )

    test.reset_index(drop=True, inplace=True)
    val_1.reset_index(drop=True, inplace=True)
    val_2.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    #if filter_negative:
    #    test_full = test_full[test_full['rating'] >= rating_threshold]
    test['test_user_idx'] = test.index
    val_1['test_user_idx'] = val_1.index
    val_2['test_user_idx'] = val_2.index

    test_dict={}
    for user, items in test_full.groupby('user_id'):
        test_dict[user] = items

    train_dict={}
    for user, items in train.groupby('user_id'):
        train_dict[user] = items
    return train, val_1, val_2, test, train_dict, test_dict

def get_users_history(test, train_dict, test_dict) -> pd.DataFrame:
    users_history = pd.DataFrame()
    for i, row in tqdm(test.iterrows()):   
        user_history = pd.concat([
            test_dict[row['user_id']].query('time_idx < @row.time_idx'), 
            train_dict[row['user_id']]
        ])
        user_history = add_time_idx(user_history)
        user_history['test_user_idx'] = [i] * len(user_history)
        users_history = pd.concat([users_history, user_history])
    return users_history
