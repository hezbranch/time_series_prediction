# split_dataset_options.py

# Input:  --input: (required) a time-series file with one or more columns of
#              role 'id'
#         --data_dict: (required) data dictionary for that file
#         --test_size: (required) fractional size of the test set, expressed as
#              a number between 0 and 1
#         --output_dir: (required) directory where output files are saved
#         --group_cols: (optional) columns to group by, specified as a
#             space-separated list
#         Additionally, a seed used for randomization is hard-coded.
# Output: train.csv and test.csv, where grouping is by all specified columns,
#         or all columns of role 'id' if --group_cols is not specified.

import argparse
import json
import pandas as pd
import os
import numpy as np
import copy

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit


class Splitter:
    def __init__(self, n_splits, size=0, random_state=0, splitter_type="", cols_to_group=None):
        self.n_splits = n_splits
        self.size = size
        self.cols_to_group = cols_to_group
        self.splitter_type = splitter_type
        if hasattr(random_state, 'rand'):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(int(random_state))

    def make_groups_from_df(self, data_df):
        grp = data_df[self.cols_to_group]
        grp = [' '.join(row) for row in grp.astype(str).values]
        return grp
        
    def split(self, X, y=None, groups=None):
        if (self.splitter_type == "group_split"):
            gss1 = GroupShuffleSplit(random_state=copy.deepcopy(self.random_state), test_size=self.size, n_splits=self.n_splits)
        elif (self.splitter_type == "stratified_split"):
            gss1 = StratifiedKFold(n_splits=self.n_splits, random_state=copy.deepcopy(self.random_state), shuffle=True)
        elif (self.splitter_type == "naive_split"):
            gss1 = KFold(n_splits=self.n_splits, random_state=copy.deepcopy(self.random_state), shuffle=True)
        else:
            gss1 = GroupShuffleSplit(random_state=copy.deepcopy(self.random_state), test_size=self.size, n_splits=self.n_splits)
            
        # Sanity check add print statements to check the first five rows
        for tr_inds, te_inds in gss1.split(X, y=y, groups=groups):
            print("RESULTS FROM TR INDS & TE INDS SPLIT HERE:")
            yield tr_inds, te_inds

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

# EXPERIMENT WITH STRATIFIED K FOLD
# TESTING SPLITS FROM 5 to 20, SPLIT MUST EXCEED 1 FOR K FOLD
# WILL NEED TO COMPARE TO RANDOM CROSS VALIDATION
def split_dataframe_by_keys(data_df=None, size=0, random_state=0, cols_to_group=None, splitter_type=""):
    # ADDING THE LINE OF CODE BELOW (HEZEKIAH)
    kfold_split = Splitter(n_splits=5, size=size, cols_to_group=cols_to_group, splitter_type=splitter_type)
    # >> for a, b in gss1.split(df, groups=gss1.make_groups_from_df(data_df)):
    for a, b in kfold_split.split(df, df['subject_id'], groups=kfold_split.make_groups_from_df(data_df)):
        train_df = df.iloc[a].copy()
        test_df = df.iloc[b].copy()
    return train_df, test_df


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--data_dict', required=True)
    parser.add_argument('--test_size', required=False, type=float)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--train_csv_filename', default='train.csv')
    parser.add_argument('--test_csv_filename', default='test.csv')
    parser.add_argument('--output_data_dict_filename', required=False, type=str, default=None)
    parser.add_argument('--group_cols', nargs='*', default=[None])
    parser.add_argument('--random_state', required=False, type=int, default=20190206)
    args = parser.parse_args()

    # Import data
    df = pd.read_csv(args.input)
    data_dict = json.load(open(args.data_dict))

    # Split dataset
    if len(args.group_cols) == 0 or args.group_cols[0] is not None:
        group_cols = args.group_cols
    elif args.group_cols[0] is None:
        try:
            fields = data_dict['fields']
        except KeyError:
            fields = data_dict['schema']['fields']
        group_cols = [c['name'] for c in fields
                      if c['role'] in ('id', 'key') and c['name'] in df.columns]

    train_df, test_df = split_dataframe_by_keys(
        df, cols_to_group=group_cols, size=args.test_size, random_state=args.random_state)

    # Write split data frames to CSV
    fdir_train_test = args.output_dir
    if fdir_train_test is not None:
        if not os.path.exists(fdir_train_test):
            os.mkdir(fdir_train_test)
        args.train_csv_filename = os.path.join(fdir_train_test, args.train_csv_filename)
        args.test_csv_filename = os.path.join(fdir_train_test, args.test_csv_filename)
        if args.output_data_dict_filename is not None:
            args.output_data_dict_filename = os.path.join(fdir_train_test, args.output_data_dict_filename)

    train_df.to_csv(args.train_csv_filename, index=False)
    test_df.to_csv(args.test_csv_filename, index=False)

    if args.output_data_dict_filename is not None:
        with open(args.output_data_dict_filename, 'w') as f:
            json.dump(data_dict, f, indent=4)


