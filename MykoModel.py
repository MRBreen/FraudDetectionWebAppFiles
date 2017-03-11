import pandas as pd
import numpy as np
import data_cleanup as dc


class MykoModel():

    def __init__(self,estimator):
        self.estimator = estimator

    def get_data(self,datafile):
        '''
        INPUT: Path of input file (JSON)
        OUTPUT: pandas df ready for modeling
        '''

        raw_data_df = pd.read_json(datafile)

        fraud_list = ['fraudster', 'fraudster_event']
        fraud = (raw_data_df.acct_type.isin(fraud_list)).astype(int)

        clean_df = self.clean_nulls_df(raw_data_df)
        model_df = self.prep_df_for_model(clean_df)

        return model_df, fraud

    def get_new_data(self,datafile):
        '''
        INPUT: Path of input file (JSON)
        OUTPUT: pandas df ready for modeling
        '''

        raw_data_df = pd.DataFrame(datafile)

        clean_df = self.clean_nulls_new_df(raw_data_df)
        model_df = self.prep_new_df_for_model(clean_df)

        return model_df

    def fit(self,X,y):
        '''
        INPUT: Training dataset (pandas), Labeled dataset
        OUTPUT: self (model that has been fitted)
        '''
        self.estimator.fit(X,y)
        return self


    def predict(self,X):
        '''
        INPUT: Test dataset (pandas)
        OUTPUT: Predicted labels
        '''
        return self.estimator.predict(X)

    def predict_proba(self,X):
       '''
       INPUT: Test dataset (pandas)
       OUTPUT: Predicted labels
       '''
       return self.estimator.predict_proba(X)

    def prep_df_for_model(self,df):
        '''
        INPUT: cleaned up df from get_clean_df()
        OUTPUT: df ready for modeling
        '''
        numeric_col_clean = df.select_dtypes(include=[np.number]).columns
        categorical_col_clean = np.setdiff1d(df.columns, numeric_col_clean)

        numeric_col_clean_mod = [
            #'approx_payout_date',
            'body_length',
            'channels',
            #'delivery_method',
            # 'event_created',
            # 'event_end',
            # 'event_published',
            # 'event_start',
            'fb_published',
            'gts',
            'has_analytics',
            'has_header',
            'has_logo',
            'name_length',
            'num_order',
            'num_payouts',
            #'object_id',
            #'org_facebook',
            #'org_twitter',
            'sale_duration',
            'sale_duration2',
            'show_map',
            'user_age',
            # 'user_created',
            'user_type',
            'venue_latitude',
            'venue_longitude',
            'delivery_method_null',
            'user_type_null',
            'org_facebook_null',
            'org_twitter_null',
            #'acct_type_null',
            'country_null',
            'currency_null',
            'description_null',
            'email_domain_null',
            'listed_null',
            'name_null',
            'event_published_null',
            'org_desc_null',
            'org_name_null',
            'payee_name_null',
            'payout_type_null',
            'previous_payouts_null',
            'ticket_types_null',
            'venue_state_null',
            # 'venue_address_null',
            # 'venue_country_null',
            'venue_name_null',
        ]

        categorical_col_clean_mod = [
            'acct_type',
            'country',
            'currency',
            'description',
            'email_domain',
            'listed',
            'name',
            'org_desc',
            'org_name',
            'payee_name',
            'payout_type',
            'previous_payouts',
            'ticket_types',
            'venue_address',
            'venue_country',
            'venue_name',
            'venue_state',
        ]

        # If the count of previous payouts is zero, probably a high risk.
        # Let's convert previous_payouts to previous_payouts_cnt (count)
        df['previous_payouts_cnt'] = df.previous_payouts.apply(lambda x: len(x))

        addl_col = [
            'previous_payouts_cnt'
        ]

        model_df = df[numeric_col_clean_mod + addl_col]

        return model_df


    def clean_nulls_df(self, df):
        '''
        INPUT: json data file
        OUTPUT: df with no nulls
        '''
        # Columns that have null values that should be replaced with 0
        cols_null_zero = [
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
        #'object_id',
        'sale_duration',
        'sale_duration2',
        'show_map',
        'user_age',
        # Consider removing these later?  Use lasso to evaluate.
        ]

        # Columns that have null values that should be replaced with mean
        cols_null_mean = [
        'venue_latitude',
        'venue_longitude',
        # 'approx_payout_date',
        # 'event_created',
        # 'event_end',
        # 'event_published',
        # 'event_start',
        # 'user_created'
        ]

        # Columns that have null values that are:
        #  1) categorical type OR
        #  2) numerical type that should be recast as categorical
        cols_null_dummy = [
        # Numerical
        'delivery_method',
        'user_type',
        'org_facebook',
        'org_twitter',
        # Categorical
        #'acct_type',
        'country',
        'currency',
        'description',
        'email_domain',
        'listed',
        'name',
        'event_published',
        'org_desc',
        'org_name',
        'payee_name',
        'payout_type',
        'previous_payouts',
        'ticket_types',
        'venue_state',
        'venue_address',
        'venue_country',
        'venue_name',
        ]

        # Clean up columns with null values, as defined above
        clean_df = dc.clean_up_nulls(df,
                                     cols_null_zero,
                                     cols_null_mean,
                                     cols_null_dummy)

        #Check for nulls
        for col in clean_df.columns:
            null_cnt = sum(clean_df[col].isnull())
            if null_cnt > 0:
                print "WARNING: {} has {} null values".format(col, null_cnt)

        return clean_df

    def prep_new_df_for_model(self,df):
        '''
        INPUT: cleaned up df from get_clean_df()
        OUTPUT: df ready for modeling
        '''
        numeric_col_clean = df.select_dtypes(include=[np.number]).columns
        categorical_col_clean = np.setdiff1d(df.columns, numeric_col_clean)

        numeric_col_clean_mod = [
            #'approx_payout_date',
            'body_length',
            'channels',
            #'delivery_method',
            # 'event_created',
            # 'event_end',
            # 'event_published',
            # 'event_start',
            'fb_published',
            'gts',
            'has_analytics',
            'has_header',
            'has_logo',
            'name_length',
            'num_order',
            'num_payouts',
            #'object_id',
            #'org_facebook',
            #'org_twitter',
            'sale_duration',
            'sale_duration2',
            'show_map',
            'user_age',
            # 'user_created',
            'user_type',
            'venue_latitude',
            'venue_longitude',
            'delivery_method_null',
            'user_type_null',
            'org_facebook_null',
            'org_twitter_null',
            #'acct_type_null',
            'country_null',
            'currency_null',
            'description_null',
            'email_domain_null',
            'listed_null',
            'name_null',
            'event_published_null',
            'org_desc_null',
            'org_name_null',
            'payee_name_null',
            'payout_type_null',
            'previous_payouts_null',
            'ticket_types_null',
            'venue_state_null',
            # 'venue_address_null',
            # 'venue_country_null',
            'venue_name_null',
        ]

        categorical_col_clean_mod = [
            'country',
            'currency',
            'description',
            'email_domain',
            'listed',
            'name',
            'org_desc',
            'org_name',
            'payee_name',
            'payout_type',
            'previous_payouts',
            'ticket_types',
            'venue_address',
            'venue_country',
            'venue_name',
            'venue_state',
        ]

        # If the count of previous payouts is zero, probably a high risk.
        # Let's convert previous_payouts to previous_payouts_cnt (count)
        df['previous_payouts_cnt'] = df.previous_payouts.apply(lambda x: len(x))

        addl_col = [
            'previous_payouts_cnt'
        ]

        model_df = df[numeric_col_clean_mod + addl_col]

        return model_df


    def clean_nulls_new_df(self, df):
        '''
        INPUT: json data file
        OUTPUT: df with no nulls
        '''
        # Columns that have null values that should be replaced with 0
        cols_null_zero = [
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
        #'object_id',
        'sale_duration',
        'sale_duration2',
        'show_map',
        'user_age',
        # Consider removing these later?  Use lasso to evaluate.
        ]

        # Columns that have null values that should be replaced with mean
        cols_null_mean = [
        'venue_latitude',
        'venue_longitude',
        # 'approx_payout_date',
        # 'event_created',
        # 'event_end',
        # 'event_published',
        # 'event_start',
        # 'user_created'
        ]

        # Columns that have null values that are:
        #  1) categorical type OR
        #  2) numerical type that should be recast as categorical
        cols_null_dummy = [
        # Numerical
        'delivery_method',
        'user_type',
        'org_facebook',
        'org_twitter',
        # Categorical
        'country',
        'currency',
        'description',
        'email_domain',
        'listed',
        'name',
        'event_published',
        'org_desc',
        'org_name',
        'payee_name',
        'payout_type',
        'previous_payouts',
        'ticket_types',
        'venue_state',
        'venue_address',
        'venue_country',
        'venue_name',
        ]

        # Clean up columns with null values, as defined above
        clean_df = dc.clean_up_nulls(df,
                                     cols_null_zero,
                                     cols_null_mean,
                                     cols_null_dummy)

        #Check for nulls
        for col in clean_df.columns:
            null_cnt = sum(clean_df[col].isnull())
            if null_cnt > 0:
                print "WARNING: {} has {} null values".format(col, null_cnt)

        return clean_df
