from pathlib import Path

import numpy as np
import pandas as pd


def prepare_data(initial_data_path: str, prepared_data_path: str, force: bool = True):
    prepared_data_path = Path(prepared_data_path)
    if prepared_data_path.exists() and not force:
        return pd.read_csv(prepared_data_path)

    columns = [
        'addr_state',
        'annual_inc',
        'delinq_2yrs',
        'dti',
        'emp_length',
        'purpose',
        'home_ownership',
        'inq_last_6mths',
        'int_rate',
        'loan_amnt',
        'open_acc',
        'total_acc',
        'acc_now_delinq',
        'revol_bal',
        'revol_util',
        'mths_since_last_delinq',
        'term',
        'funded_amnt',
        'total_pymnt',
        'loan_status',
    ]
    df = pd.read_csv(initial_data_path, usecols=columns)

    # removing current loans
    df = df[df['loan_status'].isin(('Fully Paid', 'Charged Off', 'Default'))]
    df['is_default'] = df['loan_status'] != 'Fully Paid'
    df.drop(columns=['loan_status'], inplace=True)

    # grouping US states by regions
    df['addr_state'] = fill_in_address_state(df['addr_state'])

    # filling nans
    mean_mths_since_last_delinq = df['mths_since_last_delinq'].mean()
    df['mths_since_last_delinq'].fillna(-1, inplace=True)
    mask = np.logical_and(df['delinq_2yrs'] > 0, df['mths_since_last_delinq'] < 0)
    df.loc[mask, 'mths_since_last_delinq'] = mean_mths_since_last_delinq

    df['revol_util'].fillna(df['revol_util'].median(), inplace=True)
    df['emp_length'].fillna('UNKNOWN', inplace=True)

    df.loc[df['home_ownership'] == 'ANY', 'home_ownership'] = 'OWN'

    for x in ['emp_length', 'purpose', 'addr_state', 'term', 'home_ownership']:
        df[x] = df[x].astype('category')

    df.to_csv(prepared_data_path)
    return df


def fill_in_address_state(address: pd.Series):
    west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
    south_west = ['AZ', 'TX', 'NM', 'OK']
    mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
    north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
    south_east = [
        'GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN'
    ]

    def _finding_regions(state):
        if state in west:
            return 'West'
        elif state in south_west:
            return 'SouthWest'
        elif state in south_east:
            return 'SouthEast'
        elif state in mid_west:
            return 'MidWest'
        elif state in north_east:
            return 'NorthEast'

    return address.apply(_finding_regions)
