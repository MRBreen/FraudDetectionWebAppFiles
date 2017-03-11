import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn.

def prep_data_for_model(df):

    fraud_list = ['fraudster', 'fraudster_event']
    fraud = (df.acct_type.isin(fraud_list)).astype(int)
    numeric_col = df.select_dtypes(include=[np.number]).columns

    numeric_col_clean = [
    'body_length',
    'channels',
    'fb_published',
    'gts',
    'has_analytics',
    'has_header',
    'has_logo',
    'name_length',
    'num_order',
    'num_payouts',
    # We don't need this for modeling
    #'object_id',
    'sale_duration',
    'sale_duration2',
    'show_map',
    'user_age',
    # Consider removing these later?  Use lasso to evaluate.
    'venue_latitude',
    'venue_longitude',
    ]

    numeric_col_get_dummies = [
    'delivery_method',
    'user_type',
    'org_facebook',
    'org_twitter',
    ]

    numeric_date_col = [
    'approx_payout_date'
    'event_created',
    'event_end',
    'event_published',
    'event_start',
    'user_created'
    ]

        # These are for data that non-numeric
    categorical_col_get_dummies= [
    'country',
    'payout_type',
    'email_domain',
    'delivery_method',
    'venue_state',
    'venue_country',
    'currency',
    ]
    new_df = df.copy()[numeric_col_clean]
    new_df['fraud'] = fraud
    return new_df

def clean_up_nulls(df, cols_null_zero, cols_null_mean, cols_null_dummy):
    new_df = convert_null_to_zeroes(df, cols_null_zero)
    new_df = convert_null_to_means(new_df,cols_null_mean)
    new_df = convert_null_to_dummy(new_df,cols_null_dummy)
    new_df = convert_null_to_mode(new_df,cols_null_dummy)

    return new_df

def convert_null_to_zeroes(df, cols):
    new_df = df.copy()
    for col in cols:
        new_df[col] = new_df[col].astype(str).astype(float)
        #pd.to_numeric(new_df[col], errors='coerce')
        new_df[col].fillna(0, inplace=True)
    return new_df

def convert_null_to_means(df, cols):
    new_df = df.copy()
    for col in cols:
        new_df[col] = new_df[col].astype(str).astype(float)
        col_mean = new_df[col].mean()
        #print col, col_mean
        new_df[col].fillna(col_mean, inplace=True)
    return new_df

def convert_null_to_dummy(df, cols):
    new_df = pd.DataFrame()
    for col in cols:
        new_col_name = '{}_null'.format(col)
        new_df[new_col_name] = (df[col].isnull()).astype(int)
    return pd.concat([df, new_df], axis=1)

def convert_null_to_mode(df, cols):
    new_df = df.copy()
    for col in cols:
        null_cnt = np.sum((df[col].isnull()))
        if null_cnt>0:
            new_df[col].fillna('', inplace=True)
    return new_df

# def __main__():
#     %%writefile subset_json.py
#     """head_json.py - extract a couple records from a huge json file.
#
#     Syntax: python head_json.py < infile.json > outfile.json
#     """
#
#     start_char = '{'
#     stop_char = '}'
#     n_records = num_records
#     level_nesting = 0
#
#     while n_records != 0:
#         ch = sys.stdin.read(1)
#         sys.stdout.write(ch)
#         if ch == start_char:
#             level_nesting += 1
#         if ch == stop_char:
#             level_nesting -= 1
#             if level_nesting == 0:
#                 n_records -= 1
#     sys.stdout.write(']')
