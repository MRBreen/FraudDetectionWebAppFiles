import pandas as pd

def clean_input(data_point):
    for col in data_point.keys():
        if col == u'previous_payouts':
            data_point[u'previous_payouts'] = len(data_point[u'previous_payouts'])
    df = pd.DataFrame(data_point)
    return df
